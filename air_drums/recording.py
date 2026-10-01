"""Bounded asynchronous queue for sequential, batched SQLite writes."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .strokes import initialize_database, validate_calibration_trial, validate_stroke, write_stroke_batch


LOGGER = logging.getLogger(__name__)


class RecordingQueueFull(RuntimeError):
    pass


class StrokeRecorder:
    def __init__(self, database_path: Path, legacy_path: Path | None = None, max_queue_size: int = 4096):
        self.database_path = database_path
        self.legacy_path = legacy_path
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=max_queue_size)
        self.worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await asyncio.to_thread(initialize_database, self.database_path, self.legacy_path)
        self.worker = asyncio.create_task(self._run(), name="air-drums-stroke-writer")

    async def create_session(self) -> dict[str, Any]:
        from .strokes import create_session

        await asyncio.to_thread(initialize_database, self.database_path, self.legacy_path)
        return await asyncio.to_thread(create_session, self.database_path)

    def enqueue_stroke(self, session_id: str, value: Any) -> dict[str, Any]:
        stroke = validate_stroke(value)
        try:
            self.queue.put_nowait({"kind": "stroke", "sessionId": session_id, "stroke": stroke})
        except asyncio.QueueFull as error:
            raise RecordingQueueFull("stroke recording queue is full") from error
        return stroke

    def enqueue_calibration_trial(self, session_id: str, value: Any) -> dict[str, Any]:
        trial = validate_calibration_trial(value)
        try:
            self.queue.put_nowait({"kind": "calibration", "sessionId": session_id, "trial": trial})
        except asyncio.QueueFull as error:
            raise RecordingQueueFull("stroke recording queue is full") from error
        return trial

    async def enqueue_finish(self, session_id: str) -> None:
        await self.queue.put({
            "kind": "finish",
            "sessionId": session_id,
            "endedAt": datetime.now(timezone.utc).isoformat(),
        })

    async def flush(self) -> None:
        await self.queue.join()

    async def close(self) -> None:
        await self.queue.put(None)
        if self.worker:
            await self.worker
            self.worker = None

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            if first is None:
                self.queue.task_done()
                return
            batch = [first]
            stop_after_batch = False
            # Briefly coalesce nearby strokes so a fast sequence shares one
            # transaction. Camera inference never waits on this task.
            while len(batch) < 32:
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=0.04)
                except TimeoutError:
                    break
                if item is None:
                    self.queue.task_done()
                    stop_after_batch = True
                    break
                batch.append(item)
            try:
                await asyncio.to_thread(write_stroke_batch, self.database_path, batch)
            except Exception:
                LOGGER.exception("Could not write a queued Air Drummer recording batch")
            finally:
                for _ in batch:
                    self.queue.task_done()
            if stop_after_batch:
                return
