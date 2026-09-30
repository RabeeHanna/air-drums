# Air Drummer

The app and its runtime assets are delivered through Docker. The browser opens the webcam and runs only MediaPipe Pose Landmarker Lite in a Web Worker. Pose tracks shoulders, elbows, and wrists; an estimated hand point extends from the wrist along the forearm direction. Wrist motion times each stroke, while the estimated point supplies the hit rectangle, nearest drum, and impact comparison. The estimate does not model wrist flexion. Camera frames stay in the browser and are not sent to Google; MediaPipe may send performance and utilization telemetry.

## Run

```powershell
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000), allow camera access, and select **Start camera**. The app requests a 640×480 camera stream at up to 60 FPS. **Processing resolution** selects standard input up to 640×480 or a smaller input up to 480×360; this changes the model input immediately while leaving the camera preview at its negotiated size. The actual camera and inference input sizes are shown in the metrics. Frames are processed one at a time, so slow inference does not build a backlog of stale images.

The app tries the MediaPipe GPU delegate and falls back to CPU if it cannot use the GPU. **Motion response** adjusts an adaptive One Euro display filter. Wrist landmarks use a faster setting than shoulders and elbows; low-confidence points are held at their last stable location and reacquired gradually. Higher response values reduce lag during fast movement, while lower values damp jitter. **Show raw pose points** overlays unsmoothed wrists for comparison. Metrics report camera mode, model input size, processed FPS, inference duration, and total frame delay.

The preview includes translucent oval regions for snare, hi-hat, Tom 1, Tom 2, floor tom, crash, and ride. Select **Edit layout**, drag an oval to move it, or drag one of its eight bounding-box handles to resize it. Regions may overlap; choose the active one from **Selected zone**. **Save kit** writes normalized centers and oval width/height to `data/kit-layout.json` on the host. The Docker bind mount keeps this file across container rebuilds. Existing circle settings migrate to flatter ovals while preserving their centers and horizontal reach. **Reset zones** restores the built-in layout; save to make that reset persistent.

During each downswing, the preview shows two projected impact endpoints: a short 120 ms motion estimate and an estimate based on deceleration toward the detector's impact-turn threshold. Each prediction highlights its nearest kit zone and reports an estimated time to impact. At impact, the readout compares each estimate with the detected endpoint and time.

The orange estimated hand point extends from the Pose wrist along the forearm direction. **Estimated hand length** controls the extension as a fraction of shoulder width, and **Show estimated hand point** toggles its overlay. This approximates an extended finger; it does not track wrist flexion. Stroke calibration controls adjust the wrist downward-speed threshold, minimum travel, and impact-turn sensitivity immediately. Their values are saved in this browser and can be reset to defaults. The live counters distinguish detected strokes, accepted hits, and hits outside the global hit rectangle. Drum-pad ovals assign a drum but do not gate whether a hit is counted. The hit rectangle is hidden unless it is being edited.

When **Audio hits** is enabled, an accepted estimated hand-point hit plays the nearest drum zone’s preloaded local WAV sample immediately. The app generates its seven original samples with the standard-library script `scripts/generate_drum_samples.py`; no audio service or sound package is required. Use the volume slider to adjust playback. The browser requires a user gesture to enable audio, so start the camera or turn audio back on after loading the page.

Stroke records are grouped into camera sessions in `data/air-drums.sqlite3`. A bounded in-memory queue feeds one background writer that batches nearby events into SQLite transactions; the camera and UI do not wait for disk writes. Records include timestamps, hand, estimated hand-point coordinates, hit-area acceptance, nearest actual zone, duration, peak speed, travel, and both prediction comparisons. Strokes outside the hit rectangle are retained. No video or per-frame landmark stream is stored. Older `strokes.jsonl` events are imported into one legacy session the next time the app starts; prior SQLite rows remain readable after the prediction fields are added.

Press `Ctrl+C` in the terminal to stop the app. To remove the stopped container:

```powershell
docker compose down
```

## Tests

```powershell
docker compose exec -T air-drums python -m unittest discover -s tests
docker compose run --rm stroke-tests
```

The Python tests cover kit layout persistence, API validation, session storage, and legacy-record compatibility. The Node tests cover stroke detection and both prediction models with nearest-zone comparisons. Estimated hand-point stability and endpoint accuracy still need live camera review.
