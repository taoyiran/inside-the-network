const P = require("./pageinit_node.js"), A = require("./audioharness.js"), H = require("../seeing/harness.js"), fs = require("fs"), path = require("path");
const load = (f) => JSON.parse(fs.readFileSync(path.join(__dirname, f), "utf8")).map((L) => (L.W ? { ...L, W: Float64Array.from(L.W), b: Float64Array.from(L.b) } : L));
const cnn = load("final_cnn.json"), plain = load("final_plain.json"), D = P.DATA;
const pred = (net, spec) => { const s = P.forward(net, P.toInput(spec)).at(-1).d; let b = 0; for (let j = 1; j < 10; j++) if (s[j] > s[b]) b = j; return b; };
const acc = (f) => { let c = 0, m = 0; for (let k = 0; k < D.test.n; k++) { const s = f(P.specOf(D.test, k)); if (pred(cnn, s) === D.test.labels[k]) c++; if (pred(plain, s) === D.test.labels[k]) m++; } return [(c / 10).toFixed(1), (m / 10).toFixed(1)]; };
console.log("centered: cnn / plain", acc((s) => s).join(" / "));
// closest average over the 8,000 training spectrograms
const avgs = Array.from({ length: 10 }, (_, d) => { const a = new Float64Array(1024); let n = 0; for (let k = 0; k < D.train.n; k++) if (D.train.labels[k] === d) { n++; const s = P.specOf(D.train, k); for (let i = 0; i < 1024; i++) a[i] += s.d[i]; } return a.map((v) => v / n); });
let ok = 0; for (let k = 0; k < D.test.n; k++) { const s = P.specOf(D.test, k); let b = 0, bd = Infinity; avgs.forEach((a, d) => { let e = 0; for (let i = 0; i < 1024; i++) { const t = s.d[i] - a[i]; e += t * t; } if (e < bd) { bd = e; b = d; } }); if (b === D.test.labels[k]) ok++; }
console.log("closest average:", (ok / 10).toFixed(1));
// picture shifts: columns (time, new columns silent = 0) and rows (pitch bands, new rows silent)
const shift = (s, dc, dr) => { const o = P.grid(1, 32, 32); for (let r = 0; r < 32; r++) for (let c = 0; c < 32; c++) { const sr = r + dr, sc = c - dc; o.d[r * 32 + c] = sr >= 0 && sr < 32 && sc >= 0 && sc < 32 ? s.d[sr * 32 + sc] : 0; } return o; };
for (const dc of [-4, -2, 2, 4, 6]) console.log(`later by ${dc} columns (${dc * 31} ms):`, acc((s) => shift(s, dc, 0)).join(" / "));
for (const dr of [1, 2, 3, -1, -2]) console.log(`up ${dr} bands:`, acc((s) => shift(s, 0, dr)).join(" / "));
// noise on the raw test audio
const pick = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "pick.json"), "utf8")), r = H.mulberry32(5);
const raws = pick.test.map((p) => A.readWav(path.join(__dirname, "dl", p.name)));
for (const snr of [20, 10, 5, 0]) { let c = 0, m = 0; raws.forEach((s, k) => { let e = 0; for (const v of s) e += v * v; const sd = Math.sqrt(e / s.length) / 10 ** (snr / 20), o = s.map((v) => v + sd * H.gaussian(r)); const sp = P.spectrogram(o); if (pred(cnn, sp) === D.test.labels[k]) c++; if (pred(plain, sp) === D.test.labels[k]) m++; }); console.log(`noise ${snr} dB below:`, (c / 10).toFixed(1), "/", (m / 10).toFixed(1)); }
console.log("params:", [0, 3, 6].map((l) => cnn[l].W.length + cnn[l].b.length).join(" + "), "=", [0, 3, 6].reduce((s, l) => s + cnn[l].W.length + cnn[l].b.length, 0), "| plain", plain.reduce((s, L) => s + (L.W ? L.W.length + L.b.length : 0), 0));
// band frequencies for describing "up 2 bands"
const ctr = P.MEL.map((w) => { let s = 0, m = 0; w.forEach((v, k) => { s += v * k * 16000 / 512; m += v; }); return s / m; });
console.log("2 bands up from 500 Hz band:", (ctr[8] / ctr[6]).toFixed(2), "| from 1 kHz:", (ctr[12] / ctr[10]).toFixed(2), "| from 3 kHz:", (ctr[23] / ctr[21]).toFixed(2));
