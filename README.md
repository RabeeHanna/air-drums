# Air Drummer

The app and its runtime assets are delivered through Docker. The browser opens the webcam and runs the MediaPipe Pose Landmarker Lite model in a Web Worker. Only body-pose tracking is included: the app draws shoulders, elbows, and wrists. Camera frames stay in the browser and are not sent to Google; MediaPipe may send performance and utilization telemetry.

## Run

```powershell
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000), allow camera access, and select **Start camera**. The app requests a 640×480 camera stream at up to 60 FPS. The browser may negotiate another mode based on camera support; the actual resolution and frame rate are shown with the tracking metrics. Frames are processed one at a time, so slow inference does not build a backlog of stale images.

The app tries the MediaPipe GPU delegate and falls back to CPU if it cannot use the GPU. **Motion response** adjusts an adaptive One Euro display filter. Wrist landmarks use a faster setting than shoulders and elbows; low-confidence points are held at their last stable location and reacquired gradually. Higher response values reduce wrist lag during fast movement, while lower values damp jitter. **Show raw pose points** overlays the unsmoothed model positions for comparison. Metrics report camera mode, processed FPS, inference duration, and total frame delay.

The preview includes translucent oval regions for snare, hi-hat, Tom 1, Tom 2, floor tom, crash, and ride. Select **Edit layout**, drag an oval to move it, or drag one of its eight bounding-box handles to resize it. Regions may overlap; choose the active one from **Selected zone**. **Save kit** writes normalized centers and oval width/height to `data/kit-layout.json` on the host. The Docker bind mount keeps this file across container rebuilds. Existing circle settings migrate to flatter ovals while preserving their centers and horizontal reach. **Reset zones** restores the built-in layout; save to make that reset persistent.

Press `Ctrl+C` in the terminal to stop the app. To remove the stopped container:

```powershell
docker compose down
```

## Milestone 2 review

Review whether each shaded region aligns with your mirrored playing position and whether your wrists can reach them naturally. Move and resize ovals until the arrangement feels comfortable, overlap zones where needed, and save the kit. Zone coordinates use the source-camera coordinate space internally and are mirrored in the preview.

## Tests

```powershell
docker compose exec -T air-drums python -m unittest discover -s tests
```

The tests cover the seven default zones, JSON persistence, and invalid layout rejection. Webcam reach and perspective still require a live review.
