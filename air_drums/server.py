"""Serve the Air Drummer UI and locally bundled MediaPipe assets."""

from __future__ import annotations

from pathlib import Path

from aiohttp import web


STATIC = Path(__file__).parent / "static"


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC / "index.html")


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/healthz", health)
    app.router.add_static("/assets/", STATIC / "assets", show_index=False)
    app.router.add_static("/", STATIC, show_index=False)
    return app
