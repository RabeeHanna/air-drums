const SAMPLE_PATHS = {
  snare: '/audio/snare.wav',
  hi_hat: '/audio/hi_hat.wav',
  tom_1: '/audio/tom_1.wav',
  tom_2: '/audio/tom_2.wav',
  floor_tom: '/audio/floor_tom.wav',
  crash: '/audio/crash.wav',
  ride: '/audio/ride.wav',
};

export class LocalAudioPlayer {
  constructor(onStatus = () => {}) {
    this.onStatus = onStatus;
    this.context = null;
    this.masterGain = null;
    this.buffers = new Map();
    this.voices = new Set();
    this.enabled = true;
    this.volume = 0.72;
    this.ready = false;
  }

  async prepare() {
    const AudioContextClass = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!AudioContextClass) {
      this.onStatus('Audio is not supported by this browser.');
      return false;
    }
    try {
      this.context = new AudioContextClass({ latencyHint: 'interactive' });
      this.masterGain = this.context.createGain();
      this.masterGain.gain.value = this.volume;
      this.masterGain.connect(this.context.destination);
      this.onStatus('Loading local drum sounds…');
      const loaded = await Promise.all(Object.entries(SAMPLE_PATHS).map(async ([zoneId, path]) => {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
        const buffer = await this.context.decodeAudioData(await response.arrayBuffer());
        return [zoneId, buffer];
      }));
      this.buffers = new Map(loaded);
      this.ready = true;
      this.onStatus('Local drum sounds ready. Start the camera to enable audio.');
      return true;
    } catch (error) {
      this.onStatus(`Local audio unavailable: ${error.message}`);
      return false;
    }
  }

  activate() {
    if (!this.context) return;
    this.context.resume().catch(error => this.onStatus(`Could not start audio: ${error.message}`));
  }

  setVolume(value) {
    this.volume = Math.max(0, Math.min(1, value));
    if (this.masterGain && this.context) {
      this.masterGain.gain.setTargetAtTime(this.volume, this.context.currentTime, 0.008);
    }
  }

  play(zoneId) {
    const buffer = this.buffers.get(zoneId);
    if (!this.enabled || !buffer || this.context?.state !== 'running') return false;
    if (this.voices.size >= 24) {
      const oldest = this.voices.values().next().value;
      oldest.stop();
    }
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.masterGain);
    this.voices.add(source);
    source.onended = () => this.voices.delete(source);
    source.start();
    return true;
  }
}
