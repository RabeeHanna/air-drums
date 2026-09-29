"""Session-scoped SQLite persistence for browser-detected strokes."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StrokeValidationError(ValueError):
    pass


PREDICTION_MODELS = ("fixed", "deceleration")
ZONE_IDS = {"snare", "hi_hat", "tom_1", "tom_2", "floor_tom", "crash", "ride"}


def _validate_predictions(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise StrokeValidationError("stroke predictions must be an object or null")
    predictions: dict[str, Any] = {}
    for model in PREDICTION_MODELS:
        item = value.get(model)
        if item is None:
            predictions[model] = None
            continue
        if not isinstance(item, dict):
            raise StrokeValidationError(f"{model} prediction must be an object or null")
        clean: dict[str, Any] = {}
        for field in ("x", "y", "timeToImpactMs", "predictedImpactTimestampMs", "positionErrorPx", "timingErrorMs"):
            number = item.get(field)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise StrokeValidationError(f"{model} prediction {field} must be a finite number")
            if field in ("timeToImpactMs", "positionErrorPx") and number < 0:
                raise StrokeValidationError(f"{model} prediction {field} cannot be negative")
            clean[field] = number
        zone_id = item.get("zoneId")
        if zone_id is not None and zone_id not in ZONE_IDS:
            raise StrokeValidationError(f"{model} prediction zoneId is not a known kit zone")
        clean["zoneId"] = zone_id
        predictions[model] = clean
    return predictions


def validate_stroke(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StrokeValidationError("stroke must be a JSON object")
    hand = value.get("hand")
    if hand not in ("left", "right"):
        raise StrokeValidationError("stroke hand must be left or right")
    timestamp = value.get("timestamp")
    if not isinstance(timestamp, str):
        raise StrokeValidationError("stroke timestamp must be an ISO date")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise StrokeValidationError("stroke timestamp must be an ISO date") from error
    numeric: dict[str, int | float] = {}
    for field in ("cameraTimestampMs", "durationMs", "peakSpeed", "travel"):
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise StrokeValidationError(f"stroke {field} must be a finite number")
        if number < 0:
            raise StrokeValidationError(f"stroke {field} cannot be negative")
        numeric[field] = number
    x = value.get("x")
    y = value.get("y")
    if (x is None) != (y is None):
        raise StrokeValidationError("stroke x and y must both be present or absent")
    if x is not None:
        for field, number in (("x", x), ("y", y)):
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number <= 1:
                raise StrokeValidationError(f"stroke {field} must be normalized between 0 and 1")
    inside_hit_area = value.get("insideHitArea", True)
    if not isinstance(inside_hit_area, bool):
        raise StrokeValidationError("stroke insideHitArea must be a boolean")
    actual_zone_id = value.get("actualZoneId")
    if actual_zone_id is not None and actual_zone_id not in ZONE_IDS:
        raise StrokeValidationError("stroke actualZoneId is not a known kit zone")
    return {
        "hand": hand, "timestamp": timestamp, **numeric, "x": x, "y": y,
        "insideHitArea": inside_hit_area, "actualZoneId": actual_zone_id,
        "predictions": _validate_predictions(value.get("predictions")),
    }


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT
        );
        CREATE TABLE IF NOT EXISTS strokes (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            timestamp TEXT NOT NULL,
            camera_timestamp_ms REAL NOT NULL,
            impact_x REAL,
            impact_y REAL,
            hand TEXT NOT NULL CHECK (hand IN ('left', 'right')),
            duration_ms REAL NOT NULL,
            peak_speed REAL NOT NULL,
            travel REAL NOT NULL,
            inside_hit_area INTEGER NOT NULL DEFAULT 1,
            actual_zone_id TEXT,
            predictions_json TEXT
        );
        CREATE INDEX IF NOT EXISTS strokes_by_session_time
            ON strokes(session_id, camera_timestamp_ms);
        CREATE TABLE IF NOT EXISTS imported_files (
            sha256 TEXT PRIMARY KEY,
            imported_at TEXT NOT NULL
        );
        """
    )
    stroke_columns = {row["name"] for row in connection.execute("PRAGMA table_info(strokes)")}
    if "impact_x" not in stroke_columns:
        connection.execute("ALTER TABLE strokes ADD COLUMN impact_x REAL")
    if "impact_y" not in stroke_columns:
        connection.execute("ALTER TABLE strokes ADD COLUMN impact_y REAL")
    if "inside_hit_area" not in stroke_columns:
        connection.execute("ALTER TABLE strokes ADD COLUMN inside_hit_area INTEGER NOT NULL DEFAULT 1")
    if "actual_zone_id" not in stroke_columns:
        connection.execute("ALTER TABLE strokes ADD COLUMN actual_zone_id TEXT")
    if "predictions_json" not in stroke_columns:
        connection.execute("ALTER TABLE strokes ADD COLUMN predictions_json TEXT")
    return connection


def migrate_legacy_jsonl(database_path: Path, legacy_path: Path) -> int:
    """Import the earlier flat log once, then remove it after a full import."""
    if not legacy_path.exists():
        return 0
    contents = legacy_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    connection = _connect(database_path)
    try:
        if connection.execute("SELECT 1 FROM imported_files WHERE sha256 = ?", (digest,)).fetchone():
            return 0
        records: list[dict[str, Any]] = []
        invalid_lines = False
        for line in contents.splitlines():
            try:
                records.append(validate_stroke(json.loads(line)))
            except (json.JSONDecodeError, StrokeValidationError):
                invalid_lines = True
        imported = 0
        if records:
            session_id = f"legacy-{digest[:16]}"
            times = [record["timestamp"] for record in records]
            connection.execute(
                "INSERT OR IGNORE INTO sessions(id, started_at, ended_at) VALUES (?, ?, ?)",
                (session_id, min(times), max(times)),
            )
            connection.executemany(
                """INSERT INTO strokes(session_id, timestamp, camera_timestamp_ms, hand,
                   duration_ms, peak_speed, travel, impact_x, impact_y, inside_hit_area, actual_zone_id, predictions_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(session_id, item["timestamp"], item["cameraTimestampMs"], item["hand"],
                  item["durationMs"], item["peakSpeed"], item["travel"], item["x"], item["y"],
                  int(item["insideHitArea"]), item["actualZoneId"],
                  json.dumps(item["predictions"], separators=(",", ":")) if item["predictions"] is not None else None)
                 for item in records],
            )
            imported = len(records)
        connection.execute(
            "INSERT INTO imported_files(sha256, imported_at) VALUES (?, ?)",
            (digest, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
        if not invalid_lines:
            legacy_path.unlink(missing_ok=True)
        return imported
    finally:
        connection.close()


def initialize_database(database_path: Path, legacy_path: Path | None = None) -> None:
    if legacy_path:
        migrate_legacy_jsonl(database_path, legacy_path)
    connection = _connect(database_path)
    connection.close()


def create_session(database_path: Path, legacy_path: Path | None = None) -> dict[str, Any]:
    if legacy_path:
        migrate_legacy_jsonl(database_path, legacy_path)
    session = {
        "id": str(uuid.uuid4()),
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "endedAt": None,
    }
    connection = _connect(database_path)
    try:
        connection.execute(
            "INSERT INTO sessions(id, started_at, ended_at) VALUES (?, ?, NULL)",
            (session["id"], session["startedAt"]),
        )
        connection.commit()
    finally:
        connection.close()
    return session


def write_stroke_batch(database_path: Path, records: list[dict[str, Any]]) -> None:
    connection = _connect(database_path)
    try:
        with connection:
            for record in records:
                if record["kind"] == "stroke":
                    session_id = record["sessionId"]
                    session = connection.execute(
                        "SELECT ended_at FROM sessions WHERE id = ?", (session_id,)
                    ).fetchone()
                    if session is None or session["ended_at"] is not None:
                        raise StrokeValidationError("stroke session is missing or already closed")
                    stroke = record["stroke"]
                    connection.execute(
                        """INSERT INTO strokes(session_id, timestamp, camera_timestamp_ms, hand,
                           duration_ms, peak_speed, travel, impact_x, impact_y, inside_hit_area, actual_zone_id, predictions_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (session_id, stroke["timestamp"], stroke["cameraTimestampMs"], stroke["hand"],
                         stroke["durationMs"], stroke["peakSpeed"], stroke["travel"], stroke["x"], stroke["y"],
                         int(stroke["insideHitArea"]), stroke["actualZoneId"],
                         json.dumps(stroke["predictions"], separators=(",", ":")) if stroke["predictions"] is not None else None),
                    )
                elif record["kind"] == "finish":
                    connection.execute(
                        "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                        (record["endedAt"], record["sessionId"]),
                    )
    finally:
        connection.close()


def get_session(database_path: Path, session_id: str) -> dict[str, Any] | None:
    connection = _connect(database_path)
    try:
        row = connection.execute("SELECT id, started_at, ended_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        strokes = connection.execute(
            """SELECT timestamp, camera_timestamp_ms AS cameraTimestampMs, hand,
               duration_ms AS durationMs, peak_speed AS peakSpeed, travel,
               impact_x AS x, impact_y AS y, inside_hit_area AS insideHitArea,
               actual_zone_id AS actualZoneId, predictions_json AS predictionsJson
               FROM strokes WHERE session_id = ? ORDER BY camera_timestamp_ms, id""",
            (session_id,),
        ).fetchall()
        saved_strokes = []
        for stroke in strokes:
            item = dict(stroke)
            item["insideHitArea"] = bool(item["insideHitArea"])
            item["predictions"] = json.loads(item.pop("predictionsJson")) if item["predictionsJson"] else None
            saved_strokes.append(item)
        return {
            "id": row["id"], "startedAt": row["started_at"], "endedAt": row["ended_at"],
            "strokes": saved_strokes,
        }
    finally:
        connection.close()


def list_sessions(database_path: Path, limit: int = 25) -> list[dict[str, Any]]:
    connection = _connect(database_path)
    try:
        rows = connection.execute(
            """SELECT s.id, s.started_at, s.ended_at,
               COUNT(h.id) FILTER (WHERE h.inside_hit_area = 1) AS hit_count
               FROM sessions s LEFT JOIN strokes h ON h.session_id = s.id
               GROUP BY s.id ORDER BY s.started_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {"id": row["id"], "startedAt": row["started_at"], "endedAt": row["ended_at"], "hitCount": row["hit_count"]}
            for row in rows
        ]
    finally:
        connection.close()
