import json
import tempfile
import unittest
from pathlib import Path

from air_drums.layout import (
    LayoutValidationError,
    default_layout,
    load_layout,
    save_layout,
    validate_layout,
)


class KitLayoutTests(unittest.TestCase):
    def test_default_layout_contains_requested_seven_zones(self):
        self.assertEqual(
            {zone["id"] for zone in default_layout()["zones"]},
            {"snare", "hi_hat", "tom_1", "tom_2", "floor_tom", "crash", "ride"},
        )

    def test_positions_and_sizes_round_trip_as_json(self):
        layout = default_layout()
        layout["zones"][0]["x"] = 0.42
        layout["zones"][0]["width"] = 0.31
        layout["zones"][0]["height"] = 0.17
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings" / "kit-layout.json"
            self.assertEqual(save_layout(path, layout), load_layout(path))

    def test_legacy_circle_layout_migrates_to_oval(self):
        layout = default_layout()
        layout["version"] = 1
        for zone in layout["zones"]:
            zone["radius"] = 0.1
            zone.pop("width")
            zone.pop("height")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kit-layout.json"
            path.write_text(json.dumps(layout), encoding="utf-8")
            migrated = load_layout(path)
            self.assertEqual(migrated["version"], 4)
            self.assertEqual(migrated["zones"][0]["width"], 0.2)
            self.assertAlmostEqual(migrated["zones"][0]["height"], 0.14)
            self.assertEqual(load_layout(path), migrated)

    def test_strike_boundary_is_saved_and_validated(self):
        layout = default_layout()
        layout["strikeBoundary"] = {"x": 0.41, "y": 0.62, "width": 0.62, "height": 0.434}
        self.assertEqual(validate_layout(layout)["strikeBoundary"], layout["strikeBoundary"])
        layout["strikeBoundary"]["width"] = 0.91
        with self.assertRaises(LayoutValidationError):
            validate_layout(layout)

    def test_legacy_boundary_circle_migrates_to_editable_oval(self):
        layout = default_layout()
        layout["version"] = 3
        layout["strikeBoundary"] = {"x": 0.41, "y": 0.62, "radius": 0.31}
        migrated = validate_layout(layout)
        self.assertEqual(migrated["version"], 4)
        self.assertEqual(migrated["strikeBoundary"], {
            "x": 0.41, "y": 0.62, "width": 0.62, "height": 0.434,
        })

    def test_unknown_zone_is_rejected(self):
        layout = default_layout()
        layout["zones"][0]["id"] = "kick"
        with self.assertRaises(LayoutValidationError):
            validate_layout(layout)

    def test_out_of_range_position_is_rejected(self):
        layout = default_layout()
        layout["zones"][0]["x"] = 1.2
        with self.assertRaises(LayoutValidationError):
            validate_layout(layout)


if __name__ == "__main__":
    unittest.main()
