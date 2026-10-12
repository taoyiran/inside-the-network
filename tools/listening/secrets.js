const A = require("./audioharness.js"), H = require("../seeing/harness.js"), P = require("../seeing/png.js"), fs = require("fs"), path = require("path");
const pick = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "pick.json"), "utf8"));
const load = (f) => JSON.parse(fs.readFileSync(path.join(__dirname, f), "utf8")).map((L) => (L.W ? { ...L, W: Float64Array.from(L.W), b: Float64Array.from(L.b) } : L));
const cnn = load("ref_cnn.json"), mlp = load("ref_mlp_0.1.json");
const feat = (s) => { const g = A.spectrogram(s); for (let q = 0; q < 1024; q++) g.d[q] = Math.round(g.d[q] * 15) / 15 - 0.5; return g; };
const pred = (net, g) => { const s = A.forward(net, g).at(-1).d; let b = 0; for (let j = 1; j < 10; j++) if (s[j] > s[b]) b = j; return b; };
const tests = pick.test.map((p) => ({ s: A.readWav(path.join(__dirname, "dl", p.name)), label: p.label }));
const accOf = (f) => { let c = 0, m = 0; for (const t of tests) { const g = f(t.s); if (pred(cnn, g) === t.label) c++; if (pred(mlp, g) === t.label) m++; } return [(100 * c / tests.length).toFixed(1), (100 * m / tests.length).toFixed(1)]; };
console.log("base:", accOf((s) => feat(s)).join(" / "));
// time shift: delay or advance the audio by k × 500 samples (one spectrogram column = 31 ms)
for (const k of [-4, -2, 2, 4, 6]) console.log(`time ${k > 0 ? "+" : ""}${k} columns (${k * 31} ms):`, accOf((s) => { const o = new Float64Array(16000); for (let i = 0; i < 16000; i++) { const j = i - k * 500; o[i] = j >= 0 && j < s.length ? s[j] : 0; } return feat(o); }).join(" / "));
// pitch shift in the picture: move every row up by k bands (new bottom rows are silence)
for (const k of [1, 2, 3, -2]) console.log(`picture up ${k} bands:`, accOf((s) => { const g = feat(s), o = A.grid(1, 32, 32); o.d.fill(-0.5); for (let r = 0; r < 32; r++) { const src = r + k; if (src >= 0 && src < 32) for (let c = 0; c < 32; c++) o.d[r * 32 + c] = g.d[src * 32 + c]; } return o; }).join(" / "));
// playback speed: resample (pitch and speed both change), linear interpolation
for (const f of [0.85, 0.92, 1.08, 1.15, 1.25]) console.log(`playback ×${f}:`, accOf((s) => { const o = new Float64Array(16000); const n = Math.floor(s.length / f), off = Math.floor((16000 - Math.min(16000, n)) / 2); for (let i = 0; i < Math.min(16000, n); i++) { const x = i * f, j = Math.floor(x), t = x - j; o[i + off] = (s[j] || 0) * (1 - t) + (s[j + 1] || 0) * t; } return feat(o); }).join(" / "));
// noise: white noise at a signal-to-noise ratio (dB) relative to the clip's RMS
const r = H.mulberry32(5);
for (const snr of [30, 20, 10, 5, 0]) console.log(`noise SNR ${snr} dB:`, accOf((s) => { let e = 0; for (const v of s) e += v * v; const rms = Math.sqrt(e / s.length), sd = rms / 10 ** (snr / 20), o = new Float64Array(s.length); for (let i = 0; i < s.length; i++) o[i] = s[i] + sd * H.gaussian(r); return feat(o); }).join(" / "));
// saliency pictures for a few test clips (one per digit)
const rows = [[], []];
for (let d = 0; d < 10; d++) { const t = tests.find((x) => x.label === d), g = feat(t.s); const grads = cnn.map((L) => (L.W ? { W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) } : null)); const gx = A.backward(cnn, A.forward(cnn, g), d, grads);
  rows[0].push({ data: g.d.map((v) => v + 0.5), w: 32, h: 32, kind: "gray" }); rows[1].push({ data: gx.map((v) => -v), w: 32, h: 32, kind: "div" }); }
P.tilesPNG(path.join(__dirname, "saliency.png"), rows, 4, 6);
