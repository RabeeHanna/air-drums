"""Start the Air Drummer web app."""

from __future__ import annotations

from aiohttp import web

from .server import create_app


def main() -> None:
    web.run_app(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
