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
        self.client = TestClient(TestServer(create_app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.path_patch.stop()
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


if __name__ == "__main__":
    unittest.main()
