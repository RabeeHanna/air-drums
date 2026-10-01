# Air Drummer

A webcam-based air-drumming prototype. MediaPipe Pose tracks the upper body in the browser; the app estimates a hand endpoint from the forearm, detects strokes, and plays local drum samples.

The interface includes an editable seven-pad layout, per-hand stroke controls, optional pad hit zones, tracking/performance metrics, and local session recording. The hand endpoint is estimated rather than fingertip-tracked, so hit timing and accuracy depend on camera framing, lighting, and movement.

## Run locally

Requirements: Docker Desktop with Docker Compose and a webcam. Node.js and Python do not need to be installed on the host. The first build needs an internet connection to download packages and the MediaPipe model. The app runs at `http://localhost:8000`.

```powershell
docker compose up --build
```

Allow camera access and select **Start camera**. Press `Ctrl+C` to stop; optionally run `docker compose down` to remove the container. Kit layout and session records are stored in the local `data/` folder, which is excluded from Git.

## Tests

```powershell
docker compose exec -T air-drums python -m unittest discover -s tests
docker compose run --rm stroke-tests
```
