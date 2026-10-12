// Separation prototype: a fine STFT that keeps phase, a mask, and overlap-add resynthesis.
const A = require("./audioharness.js"), P = require("../seeing/png.js"), fs = require("fs"), path = require("path");
const W = 512, HOPF = 128, BINS = W / 2 + 1, HANNW = Float64Array.from({ length: W }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / W));
const LEN = 16000, FR = Math.floor((LEN - W) / HOPF) + 1 + 4;  // frames, padded so every sample is covered
function stft(s) { const re = [], im = []; for (let t = 0; t < FR; t++) { const r = new Float64Array(W), i = new Float64Array(W); for (let n = 0; n < W; n++) { const k = t * HOPF + n - W / 2; r[n] = (k >= 0 && k < s.length ? s[k] : 0) * HANNW[n]; } A.fft(r, i); re.push(r); im.push(i); } return { re, im }; }
// inverse: conjugate trick for the inverse FFT, window again, overlap-add, divide by the summed squared windows
function istft(S, mask) { const out = new Float64Array(LEN), norm = new Float64Array(LEN);
  for (let t = 0; t < FR; t++) { const r = new Float64Array(W), i = new Float64Array(W);
    for (let k = 0; k < W; k++) { const b = k <= W / 2 ? k : W - k, m = mask ? mask[t * BINS + b] : 1; r[k] = S.re[t][k] * m; i[k] = -S.im[t][k] * m; }
    A.fft(r, i);
    for (let n = 0; n < W; n++) { const k = t * HOPF + n - W / 2; if (k >= 0 && k < LEN) { out[k] += (r[n] / W) * HANNW[n]; norm[k] += HANNW[n] * HANNW[n]; } } }
  return out.map((v, k) => (norm[k] > 1e-8 ? v / norm[k] : 0)); }
const sdr = (ref, est) => { let a = 0, b = 0; for (let i = 0; i < LEN; i++) { a += ref[i] * ref[i]; b += (ref[i] - est[i]) ** 2; } return 10 * Math.log10(a / b); };
const pick = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "pick.json"), "utf8"));
const clip = (name) => { const s = A.readWav(path.join(__dirname, "dl", name)), o = new Float64Array(LEN); o.set(s.subarray(0, LEN)); return o; };
const rms = (s) => Math.sqrt(s.reduce((a, v) => a + v * v, 0) / s.length);
const word = clip(pick.test.find((p) => p.label === 7).name), other = clip(pick.test.find((p) => p.label === 3 && p.peak > 0.3).name);
// a hum: 140 Hz with harmonics, the kind of buzz a fan or a power line makes
const hum = Float64Array.from({ length: LEN }, (_, i) => [1, 0.6, 0.45, 0.3, 0.2].reduce((a, w, h) => a + w * Math.sin((2 * Math.PI * 140 * (h + 1) * i) / 16000), 0));
const scaleTo = (s, r) => { const k = r / rms(s); return s.map((v) => v * k); };
// 1. perfect reconstruction with an all-ones mask
const Sw = stft(word); console.log("reconstruction error (max |x − istft(stft(x))|):", Math.max(...istft(Sw).map((v, i) => Math.abs(v - word[i]))).toExponential(1));
for (const [name, noise] of [["hum", scaleTo(hum, rms(word))], ["another voice", scaleTo(other, rms(word))]]) {
  const mix = word.map((v, i) => v + noise[i]), Sm = stft(mix), Sn = stft(noise), mask = new Float64Array(FR * BINS), bin = new Float64Array(FR * BINS);
  for (let t = 0; t < FR; t++) for (let b = 0; b < BINS; b++) { const ps = Sw.re[t][b] ** 2 + Sw.im[t][b] ** 2, pn = Sn.re[t][b] ** 2 + Sn.im[t][b] ** 2; mask[t * BINS + b] = ps / (ps + pn + 1e-12); bin[t * BINS + b] = ps > pn ? 1 : 0; }
  const kept = istft(Sm, mask), keptB = istft(Sm, bin);
  console.log(`${name}: word vs mix SDR ${sdr(word, mix).toFixed(1)} dB → ratio mask ${sdr(word, kept).toFixed(1)} dB, keep/remove mask ${sdr(word, keptB).toFixed(1)} dB`);
  // a hand-made mask for the hum: remove narrow bands around 140 Hz harmonics
  if (name === "hum") { const hand = new Float64Array(FR * BINS).fill(1); for (let h = 1; h <= 5; h++) { const c = Math.round((140 * h * W) / 16000); for (let b = c - 1; b <= c + 1; b++) for (let t = 0; t < FR; t++) hand[t * BINS + b] = 0; } console.log(`  painting out the 5 hum lines (3 bins each): ${sdr(word, istft(Sm, hand)).toFixed(1)} dB`); }
}
// a picture of the mix spectrogram (low 128 bins = 0–4 kHz), word + hum
const humS = scaleTo(hum, rms(word)), mixH = word.map((v, i) => v + humS[i]), Smh = stft(mixH), img = new Float64Array(FR * 128);
let top = -1e9; for (let t = 0; t < FR; t++) for (let b = 0; b < 128; b++) top = Math.max(top, 10 * Math.log10(Smh.re[t][b] ** 2 + Smh.im[t][b] ** 2 + 1e-10));
for (let t = 0; t < FR; t++) for (let b = 0; b < 128; b++) img[(127 - b) * FR + t] = Math.max(0, 1 + (10 * Math.log10(Smh.re[t][b] ** 2 + Smh.im[t][b] ** 2 + 1e-10) - top) / 70);
P.tilesPNG(path.join(__dirname, "mixspec.png"), [[{ data: img, w: FR, h: 128, kind: "gray" }]], 3, 4);
console.log("frames", FR, "bins", BINS);
