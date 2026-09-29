import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const destination = process.argv[2] || './models';
const models = [
  [
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'pose_landmarker_lite.task',
  ],
];

await mkdir(destination, { recursive: true });
for (const [url, filename] of models) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${filename}: HTTP ${response.status}`);
  await writeFile(join(destination, filename), Buffer.from(await response.arrayBuffer()));
}
