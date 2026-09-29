FROM node:22-bookworm-slim AS vision-assets

WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY scripts/download_models.mjs ./scripts/download_models.mjs
RUN mkdir -p /assets/vendor /assets/models \
    && cp node_modules/@mediapipe/tasks-vision/vision_bundle.mjs /assets/vendor/ \
    && cp -R node_modules/@mediapipe/tasks-vision/wasm /assets/vendor/wasm \
    && node scripts/download_models.mjs /assets/models

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY air_drums ./air_drums
COPY --from=vision-assets /assets ./air_drums/static/assets

EXPOSE 8000
CMD ["python", "-m", "air_drums"]
