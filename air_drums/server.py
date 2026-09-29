"""Serve the Air Drummer UI and locally bundled MediaPipe assets."""

from __future__ import annotations

import os
import asyncio
from pathlib import Path

from aiohttp import web

from .layout import LayoutValidationError, default_layout, load_layout, save_layout
from .recording import RecordingQueueFull, StrokeRecorder
from .strokes import get_session, list_sessions


STATIC = Path(__file__).parent / "static"
DATA_DIR = Path(os.environ.get("AIR_DRUMS_DATA_DIR", "/app/data"))
LAYOUT_PATH = DATA_DIR / "kit-layout.json"
STROKE_DB_PATH = DATA_DIR / "air-drums.sqlite3"
LEGACY_STROKE_LOG_PATH = DATA_DIR / "strokes.jsonl"
RECORDER = web.AppKey("stroke_recorder", StrokeRecorder)


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC / "index.html")


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def get_layout(_: web.Request) -> web.Response:
    try:
        return web.json_response(load_layout(LAYOUT_PATH))
    except LayoutValidationError as error:
        raise web.HTTPInternalServerError(text=str(error)) from error


async def get_default_layout(_: web.Request) -> web.Response:
    return web.json_response(default_layout())


async def put_layout(request: web.Request) -> web.Response:
    try:
        value = await request.json()
        layout = save_layout(LAYOUT_PATH, value)
    except (web.HTTPBadRequest, ValueError) as error:
        raise web.HTTPBadRequest(text=str(error)) from error
    return web.json_response(layout)


async def recording_lifecycle(app: web.Application):
    recorder = StrokeRecorder(STROKE_DB_PATH, LEGACY_STROKE_LOG_PATH)
    await recorder.start()
    app[RECORDER] = recorder
    try:
        yield
    finally:
        await recorder.close()


async def post_session(request: web.Request) -> web.Response:
    session = await request.app[RECORDER].create_session()
    return web.json_response(session, status=201)


async def get_sessions(request: web.Request) -> web.Response:
    recorder = request.app[RECORDER]
    await recorder.flush()
    sessions = await asyncio.to_thread(list_sessions, STROKE_DB_PATH)
    return web.json_response({"sessions": sessions})


async def get_session_details(request: web.Request) -> web.Response:
    await request.app[RECORDER].flush()
    session = await asyncio.to_thread(get_session, STROKE_DB_PATH, request.match_info["session_id"])
    if session is None:
        raise web.HTTPNotFound(text="Recording session not found")
    return web.json_response(session)


async def close_session(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    await request.app[RECORDER].enqueue_finish(session_id)
    return web.json_response({"queued": True, "id": session_id}, status=202)


async def post_stroke(request: web.Request) -> web.Response:
    try:
        value = await request.json()
        stroke = request.app[RECORDER].enqueue_stroke(request.match_info["session_id"], value)
    except (web.HTTPBadRequest, ValueError) as error:
        raise web.HTTPBadRequest(text=str(error)) from error
    except RecordingQueueFull as error:
        raise web.HTTPServiceUnavailable(text=str(error)) from error
    return web.json_response({"queued": True, "stroke": stroke}, status=202)


def create_app() -> web.Application:
    app = web.Application()
    app.cleanup_ctx.append(recording_lifecycle)
    app.router.add_get("/", index)
    app.router.add_get("/healthz", health)
    app.router.add_get("/api/kit-layout/defaults", get_default_layout)
    app.router.add_get("/api/kit-layout", get_layout)
    app.router.add_put("/api/kit-layout", put_layout)
    app.router.add_post("/api/sessions", post_session)
    app.router.add_get("/api/sessions", get_sessions)
    app.router.add_get("/api/sessions/{session_id}", get_session_details)
    app.router.add_post("/api/sessions/{session_id}/finish", close_session)
    app.router.add_post("/api/sessions/{session_id}/strokes", post_stroke)
    app.router.add_static("/assets/", STATIC / "assets", show_index=False)
    app.router.add_static("/", STATIC, show_index=False)
    return app
