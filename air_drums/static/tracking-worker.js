import { FilesetResolver, PoseLandmarker } from '/assets/vendor/vision_bundle.mjs';

const WASM_PATH = new URL('/assets/vendor/wasm/', self.location.origin).href;
const POSE_MODEL = '/assets/models/pose_landmarker_lite.task';

let generation = 0;
let moduleInstance = 0;
let delegate = 'CPU';
let vision = null;
let poseLandmarker = null;

function closeTask() {
  poseLandmarker?.close();
  poseLandmarker = null;
}

async function createTask(requestedDelegate) {
  // The app runs in an ES module worker, so it must use MediaPipe's matching
  // ES module WASM loader rather than the classic loader's global ModuleFactory.
  vision ??= await FilesetResolver.forVisionTasks(WASM_PATH, true);
  // The ES-module loader installs a one-shot global ModuleFactory. Give each
  // task a unique module URL so loading it again is not suppressed by the
  // browser module cache after the previous task consumes that factory.
  const createFileset = () => ({
    ...vision,
    wasmLoaderPath: `${vision.wasmLoaderPath}?instance=${++moduleInstance}`,
  });
  poseLandmarker = await PoseLandmarker.createFromOptions(createFileset(), {
    baseOptions: { modelAssetPath: POSE_MODEL, delegate: requestedDelegate },
    runningMode: 'VIDEO',
    numPoses: 1,
    outputSegmentationMasks: false,
    minPoseDetectionConfidence: 0.45,
    minPosePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
  });
}

async function initialize(data) {
  generation = data.generation;
  closeTask();

  if (data.delegatePreference === 'CPU') {
    try {
      await createTask('CPU');
      delegate = 'CPU';
      self.postMessage({ type: 'ready', generation, delegate, fallback: false });
    } catch (error) {
      closeTask();
      self.postMessage({
        type: 'fatal',
        generation,
        message: `Could not initialize MediaPipe on CPU: ${error?.message || error}`,
      });
    }
    return;
  }

  try {
    await createTask('GPU');
    delegate = 'GPU';
    self.postMessage({ type: 'ready', generation, delegate, fallback: false });
  } catch (gpuError) {
    closeTask();
    try {
      await createTask('CPU');
      delegate = 'CPU';
      self.postMessage({
        type: 'ready',
        generation,
        delegate,
        fallback: true,
        fallbackReason: String(gpuError?.message || gpuError).slice(0, 180),
      });
    } catch (cpuError) {
      closeTask();
      self.postMessage({
        type: 'fatal',
        generation,
        message: `Could not initialize MediaPipe: ${cpuError?.message || cpuError}`,
      });
    }
  }
}

function normalizedPoint(point) {
  return {
    x: Math.max(0, Math.min(1, point.x)),
    y: Math.max(0, Math.min(1, point.y)),
    z: point.z ?? 0,
    visibility: Math.max(0, Math.min(1, Math.min(point.visibility ?? 1, point.presence ?? 1))),
  };
}

function serializePose(result) {
  const landmarks = result?.landmarks?.[0];
  if (!landmarks) return null;
  return {
    leftShoulder: normalizedPoint(landmarks[11]),
    rightShoulder: normalizedPoint(landmarks[12]),
    leftElbow: normalizedPoint(landmarks[13]),
    rightElbow: normalizedPoint(landmarks[14]),
    leftWrist: normalizedPoint(landmarks[15]),
    rightWrist: normalizedPoint(landmarks[16]),
  };
}

async function processFrame(data) {
  const bitmap = data.bitmap;
  if (data.generation !== generation || !poseLandmarker) {
    bitmap.close();
    return;
  }

  try {
    const started = performance.now();
    let pose;
    try {
      const poseResult = poseLandmarker.detectForVideo(bitmap, data.timestampMs);
      pose = serializePose(poseResult);
    } catch (gpuError) {
      if (delegate !== 'GPU') throw gpuError;
      closeTask();
      await createTask('CPU');
      delegate = 'CPU';
      self.postMessage({
        type: 'delegate-fallback',
        generation,
        delegate,
        reason: String(gpuError?.message || gpuError).slice(0, 180),
      });
      const poseResult = poseLandmarker.detectForVideo(bitmap, data.timestampMs);
      pose = serializePose(poseResult);
    }
    self.postMessage({
      type: 'result',
      generation,
      timestampMs: data.timestampMs,
      delegate,
      pose,
      inference: { totalMs: performance.now() - started, width: bitmap.width, height: bitmap.height },
    });
  } catch (error) {
    self.postMessage({
      type: 'detection-error',
      generation,
      message: String(error?.message || error).slice(0, 220),
    });
  } finally {
    bitmap.close();
  }
}

self.onmessage = async event => {
  if (event.data.type === 'initialize') await initialize(event.data);
  else if (event.data.type === 'frame') await processFrame(event.data);
};
