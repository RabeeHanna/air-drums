"""Generate small original drum WAV samples using only the Python standard library."""

from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path


RATE = 24_000
OUTPUT = Path(__file__).resolve().parents[1] / "air_drums" / "static" / "audio"


def tone(t: float, frequency: float) -> float:
    return math.sin(2 * math.pi * frequency * t)


def snare(t: float, noise: float) -> float:
    envelope = math.exp(-t * 17)
    high_pass_noise = noise - 0.82 * math.sin(2 * math.pi * 230 * t)
    return 0.65 * high_pass_noise * envelope + 0.42 * tone(t, 185) * math.exp(-t * 23)


def tom(start: float, end: float):
    def sample(t: float, _noise: float = 0) -> float:
        frequency = end + (start - end) * math.exp(-t * 9)
        return tone(t, frequency) * math.exp(-t * 8)
    return sample


def cymbal(decay: float, brightness: float):
    def sample(t: float, noise: float) -> float:
        envelope = math.exp(-t * decay)
        metallic = (
            tone(t, 3100) + 0.7 * tone(t, 4870) + 0.5 * tone(t, 6310)
        ) / 2.2
        high_noise = noise - 0.78 * tone(t, 900)
        return envelope * (brightness * high_noise + (1 - brightness) * metallic)
    return sample


def write_sample(name: str, duration: float, synth, seed: int) -> None:
    rng = random.Random(seed)
    values = []
    for index in range(round(RATE * duration)):
        t = index / RATE
        noise = rng.uniform(-1, 1)
        values.append(synth(t, noise))
    peak = max(abs(value) for value in values) or 1
    scale = 0.88 / peak
    pcm = b"".join(struct.pack("<h", round(max(-1, min(1, value * scale)) * 32767)) for value in values)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUTPUT / f"{name}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm)


def main() -> None:
    samples = {
        "snare": (0.28, snare),
        "hi_hat": (0.13, cymbal(36, 0.88)),
        "tom_1": (0.36, tom(190, 92)),
        "tom_2": (0.39, tom(150, 72)),
        "floor_tom": (0.48, tom(112, 49)),
        "crash": (0.85, cymbal(3.8, 0.72)),
        "ride": (0.72, cymbal(5.5, 0.48)),
    }
    for seed, (name, (duration, synth)) in enumerate(samples.items(), start=10):
        write_sample(name, duration, synth, seed)


if __name__ == "__main__":
    main()
