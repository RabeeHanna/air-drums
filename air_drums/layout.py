"""Default kit geometry and JSON persistence for the browser kit editor."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_ZONES: tuple[dict[str, Any], ...] = (
    {"id": "snare", "label": "Snare", "x": 0.50, "y": 0.70, "width": 0.24, "height": 0.16, "color": "#f5cb5c"},
    # x/y are raw camera coordinates; CSS mirrors the preview, so the
    # conventional player's-left kit side uses a higher raw x value.
    {"id": "hi_hat", "label": "Hi-hat", "x": 0.72, "y": 0.69, "width": 0.23, "height": 0.15, "color": "#61aaff"},
    {"id": "tom_1", "label": "Tom 1", "x": 0.59, "y": 0.47, "width": 0.21, "height": 0.14, "color": "#43e0b5"},
    {"id": "tom_2", "label": "Tom 2", "x": 0.41, "y": 0.47, "width": 0.21, "height": 0.14, "color": "#43e0b5"},
    {"id": "floor_tom", "label": "Floor tom", "x": 0.27, "y": 0.69, "width": 0.24, "height": 0.16, "color": "#43e0b5"},
    {"id": "crash", "label": "Crash", "x": 0.73, "y": 0.28, "width": 0.25, "height": 0.15, "color": "#fd7ee2"},
    {"id": "ride", "label": "Ride", "x": 0.27, "y": 0.28, "width": 0.25, "height": 0.16, "color": "#fd7ee2"},
)
ZONE_IDS = frozenset(zone["id"] for zone in DEFAULT_ZONES)
DEFAULT_STRIKE_BOUNDARY = {"x": 0.5, "y": 0.55, "width": 0.84, "height": 0.588}


class LayoutValidationError(ValueError):
    """Raised when a submitted kit layout is invalid."""


def default_layout() -> dict[str, Any]:
    return {
        "version": 4,
        "zones": [dict(zone) for zone in DEFAULT_ZONES],
        "strikeBoundary": dict(DEFAULT_STRIKE_BOUNDARY),
    }


def _number(value: Any, field: str, zone_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LayoutValidationError(f"{zone_id}.{field} must be a finite number")
    return float(value)


def validate_layout(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") not in (1, 2, 3, 4):
        raise LayoutValidationError("layout must be an object with version 1, 2, 3, or 4")
    legacy_circles = value["version"] == 1
    zones = value.get("zones")
    if not isinstance(zones, list) or len(zones) != len(DEFAULT_ZONES):
        raise LayoutValidationError("layout must contain exactly seven zones")

    by_id: dict[str, dict[str, Any]] = {}
    for zone in zones:
        if not isinstance(zone, dict) or not isinstance(zone.get("id"), str):
            raise LayoutValidationError("each zone must have an id")
        zone_id = zone["id"]
        if zone_id not in ZONE_IDS or zone_id in by_id:
            raise LayoutValidationError(f"unknown or duplicate zone id: {zone_id}")
        x = _number(zone.get("x"), "x", zone_id)
        y = _number(zone.get("y"), "y", zone_id)
        if legacy_circles:
            radius = _number(zone.get("radius"), "radius", zone_id)
            width = radius * 2
            # Keep the old circle's horizontal reach while making a more
            # natural camera-space oval for the new editor.
            height = radius * 1.4
        else:
            width = _number(zone.get("width"), "width", zone_id)
            height = _number(zone.get("height"), "height", zone_id)
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise LayoutValidationError(f"{zone_id} position must be normalized between 0 and 1")
        if not 0.06 <= width <= 0.70:
            raise LayoutValidationError(f"{zone_id} width must be between 0.06 and 0.70")
        if not 0.06 <= height <= 0.50:
            raise LayoutValidationError(f"{zone_id} height must be between 0.06 and 0.50")
        by_id[zone_id] = {"x": x, "y": y, "width": width, "height": height}

    if by_id.keys() != ZONE_IDS:
        raise LayoutValidationError("layout is missing one or more kit zones")

    zones_out = []
    for default in DEFAULT_ZONES:
        zones_out.append({**default, **by_id[default["id"]]})
    boundary = value.get("strikeBoundary", DEFAULT_STRIKE_BOUNDARY)
    if not isinstance(boundary, dict):
        raise LayoutValidationError("strikeBoundary must be an object")
    boundary_x = _number(boundary.get("x"), "x", "strikeBoundary")
    boundary_y = _number(boundary.get("y"), "y", "strikeBoundary")
    if value["version"] < 4:
        boundary_radius = _number(boundary.get("radius", 0.42), "radius", "strikeBoundary")
        boundary_width = boundary_radius * 2
        boundary_height = boundary_radius * 1.4
    else:
        boundary_width = _number(boundary.get("width"), "width", "strikeBoundary")
        boundary_height = _number(boundary.get("height"), "height", "strikeBoundary")
    if not 0 <= boundary_x <= 1 or not 0 <= boundary_y <= 1:
        raise LayoutValidationError("strikeBoundary position must be normalized between 0 and 1")
    if not 0.08 <= boundary_width <= 0.90:
        raise LayoutValidationError("strikeBoundary width must be between 0.08 and 0.90")
    if not 0.08 <= boundary_height <= 0.70:
        raise LayoutValidationError("strikeBoundary height must be between 0.08 and 0.70")
    return {
        "version": 4,
        "zones": zones_out,
        "strikeBoundary": {"x": boundary_x, "y": boundary_y, "width": boundary_width, "height": boundary_height},
    }


def save_layout(path: Path, value: Any) -> dict[str, Any]:
    layout = validate_layout(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(layout, temporary, indent=2)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return layout


def load_layout(path: Path) -> dict[str, Any]:
    if not path.exists():
        return save_layout(path, default_layout())
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        layout = validate_layout(value)
        if value.get("version") != 4:
            return save_layout(path, layout)
        return layout
    except (OSError, json.JSONDecodeError, LayoutValidationError) as error:
        raise LayoutValidationError(f"Could not read kit layout: {error}") from error
