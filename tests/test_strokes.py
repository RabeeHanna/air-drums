import sqlite3
import tempfile
import unittest
from pathlib import Path

from air_drums.strokes import get_session, initialize_database


class StrokeDatabaseMigrationTests(unittest.TestCase):
    def test_old_session_rows_remain_readable_after_prediction_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "old.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.executescript("""
                CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT);
                CREATE TABLE strokes (
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    timestamp TEXT NOT NULL, camera_timestamp_ms REAL NOT NULL,
                    impact_x REAL, impact_y REAL, hand TEXT NOT NULL,
                    duration_ms REAL NOT NULL, peak_speed REAL NOT NULL, travel REAL NOT NULL
                );
                INSERT INTO sessions VALUES ('old-session', '2026-09-29T12:00:00Z', NULL);
                INSERT INTO strokes(session_id, timestamp, camera_timestamp_ms, impact_x, impact_y,
                    hand, duration_ms, peak_speed, travel)
                    VALUES ('old-session', '2026-09-29T12:00:01Z', 1000, 0.4, 0.6, 'left', 40, 2.5, 0.1);
            """)
            connection.commit()
            connection.close()

            initialize_database(database_path)
            session = get_session(database_path, "old-session")

        self.assertEqual(len(session["strokes"]), 1)
        self.assertIsNone(session["strokes"][0]["predictions"])
        self.assertTrue(session["strokes"][0]["insideHitArea"])


if __name__ == "__main__":
    unittest.main()
