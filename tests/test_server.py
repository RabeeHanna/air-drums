import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from air_drums.layout import default_layout
from air_drums.server import create_app


class KitLayoutAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.layout_path = Path(self.temporary_directory.name) / "kit-layout.json"
        self.path_patch = patch("air_drums.server.LAYOUT_PATH", self.layout_path)
        self.path_patch.start()
        self.stroke_database = Path(self.temporary_directory.name) / "recordings.sqlite3"
        self.stroke_db_patch = patch("air_drums.server.STROKE_DB_PATH", self.stroke_database)
        self.stroke_db_patch.start()
        self.legacy_path_patch = patch(
            "air_drums.server.LEGACY_STROKE_LOG_PATH", Path(self.temporary_directory.name) / "legacy.jsonl"
        )
        self.legacy_path_patch.start()
        self.client = TestClient(TestServer(create_app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.path_patch.stop()
        self.stroke_db_patch.stop()
        self.legacy_path_patch.stop()
        self.temporary_directory.cleanup()

    async def test_get_initializes_defaults_and_put_persists_changes(self):
        response = await self.client.get("/api/kit-layout")
        self.assertEqual(response.status, 200)
        layout = await response.json()
        self.assertEqual(len(layout["zones"]), 7)
        self.assertTrue(self.layout_path.exists())

        layout["zones"][0]["x"] = 0.43
        response = await self.client.put("/api/kit-layout", json=layout)
        self.assertEqual(response.status, 200)
        response = await self.client.get("/api/kit-layout")
        saved = await response.json()
        self.assertEqual(saved["zones"][0]["x"], 0.43)

    async def test_invalid_layout_and_json_are_rejected_without_overwrite(self):
        original = default_layout()
        await self.client.put("/api/kit-layout", json=original)
        invalid = default_layout()
        invalid["zones"][0]["x"] = 2

        response = await self.client.put("/api/kit-layout", json=invalid)
        self.assertEqual(response.status, 400)
        response = await self.client.put(
            "/api/kit-layout", data="{", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status, 400)

        response = await self.client.get("/api/kit-layout")
        self.assertEqual(await response.json(), original)

    async def test_strokes_are_queued_and_grouped_in_sqlite_sessions(self):
        stroke = {
            "hand": "left", "timestamp": "2026-09-29T12:00:00.000Z",
            "cameraTimestampMs": 1200, "durationMs": 132, "peakSpeed": 3.5, "travel": 0.2,
            "x": 0.48, "y": 0.63,
        }
        response = await self.client.post("/api/sessions")
        self.assertEqual(response.status, 201)
        session_id = (await response.json())["id"]
        response = await self.client.post(f"/api/sessions/{session_id}/strokes", json=stroke)
        self.assertEqual(response.status, 202)
        response = await self.client.post(f"/api/sessions/{session_id}/strokes", json={**stroke, "hand": "middle"})
        self.assertEqual(response.status, 400)
        response = await self.client.post(f"/api/sessions/{session_id}/finish")
        self.assertEqual(response.status, 202)
        response = await self.client.get(f"/api/sessions/{session_id}")
        saved = await response.json()
        self.assertEqual(len(saved["strokes"]), 1)
        self.assertEqual(saved["strokes"][0]["hand"], "left")
        self.assertEqual(saved["strokes"][0]["x"], 0.48)
        self.assertIsNotNone(saved["endedAt"])
        self.assertTrue(self.stroke_database.exists())

    async def test_prediction_comparisons_and_outside_strokes_are_persisted(self):
        stroke = {
            "hand": "right", "timestamp": "2026-09-29T12:01:00.000Z",
            "cameraTimestampMs": 1300, "durationMs": 85, "peakSpeed": 5.1, "travel": 0.16,
            "x": 0.91, "y": 0.72, "insideHitArea": False, "actualZoneId": "ride",
            "predictions": {
                "fixed": {
                    "x": 0.86, "y": 0.68, "timeToImpactMs": 120,
                    "predictedImpactTimestampMs": 1420, "zoneId": "ride",
                    "positionErrorPx": 36.0, "timingErrorMs": 45.0,
                },
                "deceleration": None,
            },
        }
        session_id = (await (await self.client.post("/api/sessions")).json())["id"]
        response = await self.client.post(f"/api/sessions/{session_id}/strokes", json=stroke)
        self.assertEqual(response.status, 202)
        await self.client.post(f"/api/sessions/{session_id}/finish")
        saved = await (await self.client.get(f"/api/sessions/{session_id}")).json()
        self.assertEqual(saved["strokes"][0]["insideHitArea"], False)
        self.assertEqual(saved["strokes"][0]["actualZoneId"], "ride")
        self.assertEqual(saved["strokes"][0]["predictions"]["fixed"]["timingErrorMs"], 45.0)
        self.assertIsNone(saved["strokes"][0]["predictions"]["deceleration"])
        sessions = await (await self.client.get("/api/sessions")).json()
        self.assertEqual(sessions["sessions"][0]["hitCount"], 0)

    async def test_legacy_flat_log_is_imported_once_into_database(self):
        legacy_path = Path(self.temporary_directory.name) / "legacy.jsonl"
        legacy_path.write_text(
            '{"hand":"right","timestamp":"2026-09-29T12:00:00Z","cameraTimestampMs":10,"durationMs":20,"peakSpeed":2,"travel":0.1}\n',
            encoding="utf-8",
        )
        # App startup performed the migration before the fixture was created;
        # a new session triggers the one-time compatibility import.
        await self.client.post("/api/sessions")
        self.assertFalse(legacy_path.exists())
        response = await self.client.get("/api/sessions")
        result = await response.json()
        self.assertTrue(any(row["hitCount"] == 1 for row in result["sessions"]))


if __name__ == "__main__":
    unittest.main()
