# Air Drummer

The app and its runtime assets are delivered through Docker. The browser opens the webcam and runs the MediaPipe Pose Landmarker Lite model in a Web Worker. Only body-pose tracking is included: the app draws shoulders, elbows, and wrists. Camera frames stay in the browser and are not sent to Google; MediaPipe may send performance and utilization telemetry.

## Run

```powershell
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000), allow camera access, and select **Start camera**. The app requests a 640×480 camera stream at up to 60 FPS. The browser may negotiate another mode based on camera support; the actual resolution and frame rate are shown with the tracking metrics. Frames are processed one at a time, so slow inference does not build a backlog of stale images.

The app tries the MediaPipe GPU delegate and falls back to CPU if it cannot use the GPU. **Motion response** adjusts an adaptive One Euro display filter. Wrist landmarks use a faster setting than shoulders and elbows; low-confidence points are held at their last stable location and reacquired gradually. Higher response values reduce wrist lag during fast movement, while lower values damp jitter. **Show raw pose points** overlays the unsmoothed model positions for comparison. Metrics report camera mode, processed FPS, inference duration, and total frame delay.

Press `Ctrl+C` in the terminal to stop the app. To remove the stopped container:

```powershell
docker compose down
```

## Tracking review

Check that both wrists stay visible when the arms extend, cross, and move quickly. Compare the raw and filtered markers and adjust Motion response. More light can help reduce webcam motion blur; high capture FPS alone cannot undo blur already present in a frame.
