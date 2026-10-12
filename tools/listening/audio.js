// Sound, as a computer gets it: a list of numbers between −1 and 1, here 16,000 of them per second.
// A spectrogram turns that list into a picture: time runs from left to right, pitch from bottom to top.
const RATE = 16000, WIN = 512, HOP = 500, FRAMES = 32, BANDS = 32;
// QUIET: a recording whose loudest point is quieter than this (in decibels) is measured from QUIET instead, so near-silence
// is not stretched to fill the whole range. Every recording here is louder.
const QUIET = -20;

// The Fourier transform of one short window says how strongly each frequency (pitch) is present in it.
// This is the fast version, the FFT: it reorders the samples, then combines pairs, then pairs of pairs.
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2, step = (-2 * Math.PI) / size;
    for (let start = 0; start < n; start += size) for (let k = 0; k < half; k++) {
      const c = Math.cos(step * k), s = Math.sin(step * k), a = start + k, b = a + half;
      const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}

// Each window fades in and out (a Hann window), so its sudden edges add far fewer frequencies that are not there.
// We hear pitch roughly on a log scale, so the 32 bands are narrow at low frequencies and wide at high ones
// (the mel scale). Each band adds up the power of the frequencies under a triangle.
const HANN = Float64Array.from({ length: WIN }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / WIN));
const toMel = (f) => 2595 * Math.log10(1 + f / 700), fromMel = (m) => 700 * (10 ** (m / 2595) - 1);
const MEL = Array.from({ length: BANDS }, (_, b) => {
  const top = toMel(RATE / 2), lo = fromMel((b * top) / (BANDS + 1)), mid = fromMel(((b + 1) * top) / (BANDS + 1)), hi = fromMel(((b + 2) * top) / (BANDS + 1));
  return Float64Array.from({ length: WIN / 2 + 1 }, (_, k) => { const f = (k * RATE) / WIN; return f <= lo || f >= hi ? 0 : f < mid ? (f - lo) / (mid - lo) : (hi - f) / (hi - mid); });
});

// The spectrogram of one second: for each of 32 moments, window the sound, find the power of every frequency,
// add it up in the 32 bands, and take the logarithm in decibels (we hear loudness on a log scale too).
// Each value is then measured from the loudest point of the clip, so a quiet and a loud "seven" look alike:
// 0 dB below the loudest point becomes 1, 80 dB or more below becomes 0, in 16 levels. The top row is the highest band.
function spectrogram(samples) {
  const out = grid(1, BANDS, FRAMES), re = new Float64Array(WIN), im = new Float64Array(WIN), db = new Float64Array(BANDS * FRAMES);
  for (let t = 0; t < FRAMES; t++) {
    for (let n = 0; n < WIN; n++) { const i = t * HOP + n; re[n] = (i < samples.length ? samples[i] : 0) * HANN[n]; im[n] = 0; }
    fft(re, im);
    for (let b = 0; b < BANDS; b++) {
      let power = 0;
      MEL[b].forEach((w, k) => { if (w) power += w * (re[k] * re[k] + im[k] * im[k]); });
      db[(BANDS - 1 - b) * FRAMES + t] = 10 * Math.log10(power + 1e-10);
    }
  }
  const loudest = Math.max(QUIET, ...db);
  db.forEach((v, i) => { out.d[i] = Math.round(15 * Math.max(0, 1 + (v - loudest) / 80)) / 15; });
  return out;
}
