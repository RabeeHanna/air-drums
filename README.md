# Air Drummer

The app and its runtime assets are delivered through Docker. The browser opens the webcam and runs the MediaPipe Pose Landmarker Lite model in a Web Worker. Only body-pose tracking is included: the app draws shoulders, elbows, and wrists. Camera frames stay in the browser and are not sent to Google; MediaPipe may send performance and utilization telemetry.

## Run

```powershell
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000), allow camera access, and select **Start camera**. The app requests a 640×480 camera stream at up to 60 FPS. The browser may negotiate another mode based on camera support; the actual resolution and frame rate are shown with the tracking metrics. Frames are processed one at a time, so slow inference does not build a backlog of stale images.

The app tries the MediaPipe GPU delegate and falls back to CPU if it cannot use the GPU. **Motion response** adjusts an adaptive One Euro display filter. Wrist landmarks use a faster setting than shoulders and elbows; low-confidence points are held at their last stable location and reacquired gradually. Higher response values reduce wrist lag during fast movement, while lower values damp jitter. **Show raw pose points** overlays the unsmoothed model positions for comparison. Metrics report camera mode, processed FPS, inference duration, and total frame delay.

The preview includes translucent oval regions for snare, hi-hat, Tom 1, Tom 2, floor tom, crash, and ride. Select **Edit layout**, drag an oval to move it, or drag one of its eight bounding-box handles to resize it. Regions may overlap; choose the active one from **Selected zone**. **Save kit** writes normalized centers and oval width/height to `data/kit-layout.json` on the host. The Docker bind mount keeps this file across container rebuilds. Existing circle settings migrate to flatter ovals while preserving their centers and horizontal reach. **Reset zones** restores the built-in layout; save to make that reset persistent.

During each downswing, the preview shows two projected impact endpoints: a short 120 ms motion estimate and an estimate based on deceleration toward the detector's impact-turn threshold. Each prediction highlights its nearest kit zone and reports an estimated time to impact. At impact, the readout compares each estimate with the detected endpoint and time. These are experimental M4 baselines; review them in free play before using intended-drum prompts in M5.

Stroke records are grouped into camera sessions in `data/air-drums.sqlite3`. A bounded in-memory queue feeds one background writer that batches nearby events into SQLite transactions; the camera and UI do not wait for disk writes. Records include timestamps, hand, impact point, hit-area acceptance, nearest actual zone, duration, peak speed, travel, and both prediction comparisons. Strokes outside the hit rectangle are retained. No video or per-frame landmark stream is stored. Older `strokes.jsonl` events are imported into one legacy session the next time the app starts; prior SQLite rows remain readable after the prediction fields are added.

Press `Ctrl+C` in the terminal to stop the app. To remove the stopped container:

```powershell
docker compose down
```

## Milestone 4 review

With the camera running, try quick and ordinary downward strokes, crossed hands, alternating hands, and pauses between hits. Compare the cyan fixed-horizon and pink impact-turn markers with the detected endpoint, zone, and timing summary. Check whether the two estimates stay stable during a downswing and whether one consistently gives a closer endpoint and time. Free play does not ask for an intended drum; prompted target accuracy is part of M5. See [MILESTONES.md](MILESTONES.md) for checkpoint status and the remaining plan.

## Tests

```powershell
docker compose exec -T air-drums python -m unittest discover -s tests
docker compose run --rm stroke-tests
```

The Python tests cover kit layout persistence, API validation, session storage, and legacy-record compatibility. The Node tests cover downswing/impact/rebound transitions and both prediction models, including invalid tracking and nearest-zone comparisons. Prediction quality still requires a live review.
