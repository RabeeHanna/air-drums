"""Serve the Air Drummer UI and locally bundled MediaPipe assets."""

from __future__ import annotations

import os
from pathlib import Path

from aiohttp import web

from .layout import LayoutValidationError, default_layout, load_layout, save_layout


STATIC = Path(__file__).parent / "static"
DATA_DIR = Path(os.environ.get("AIR_DRUMS_DATA_DIR", "/app/data"))
LAYOUT_PATH = DATA_DIR / "kit-layout.json"


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


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/healthz", health)
    app.router.add_get("/api/kit-layout/defaults", get_default_layout)
    app.router.add_get("/api/kit-layout", get_layout)
    app.router.add_put("/api/kit-layout", put_layout)
    app.router.add_static("/assets/", STATIC / "assets", show_index=False)
    app.router.add_static("/", STATIC, show_index=False)
    return app
