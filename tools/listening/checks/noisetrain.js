// Does training with hiss added make this network more robust? Half of each batch gets white noise at a random 0–20 dB below the clip RMS.
const P = require("../pagecode5p.js"), A = require("../audioharness.js"), H = require("../../seeing/harness.js"), fs = require("fs"), path = require("path");
const pick = JSON.parse(fs.readFileSync(path.join(__dirname, "../data/pick.json"), "utf8")), D = P.DATA;
const load = (list) => list.map((p) => { const s = A.readWav(path.join(__dirname, "../dl", p.name)); return Float32Array.from(s); });
const train = load(pick.train), test = load(pick.test);
const rmsOf = (s) => { let e = 0; for (const v of s) e += v * v; return Math.sqrt(e / s.length); };
const net = P.makeCNN(1), r = P.mulberry32(7), ra = P.mulberry32(1234);
let order = null, pos = 0, pass = 0;
for (let step = 0; step < 4000; step++) {
  if (!order || pos >= order.length) { order = Array.from({ length: D.train.n }, (_, i) => i); for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } pos = 0; pass++; }
  const batch = order.slice(pos, pos + 20).map((k) => {
    if (ra() < 0.5) return { image: P.toInput(P.specOf(D.train, k)), label: D.train.labels[k] };
    const s = train[k], sd = rmsOf(s) / 10 ** ((20 * ra()) / 20), o = Float64Array.from(s, (v) => v + sd * P.gaussian(ra));
    return { image: P.toInput(P.spectrogram(o)), label: D.train.labels[k] };
  });
  pos += 20; P.trainStep(net, batch, pass <= 8 ? 0.1 : 0.03);
}
const acc = (nt, f) => { let c = 0; for (let k = 0; k < D.test.n; k++) if (P.argmax(P.probsOf(nt, f(k))) === D.test.labels[k]) c++; return (c / 10).toFixed(1); };
const line = [`clean ${acc(net, (k) => P.specOf(D.test, k))}`];
for (const snr of [20, 10, 5, 0]) { const rr = H.mulberry32(5); const specs = test.map((s) => { const sd = rmsOf(s) / 10 ** (snr / 20); return P.spectrogram(Float64Array.from(s, (v) => v + sd * H.gaussian(rr))); }); line.push(`${snr} dB: ${acc(net, (k) => specs[k])}`); }
console.log("noise-trained cnn:", line.join(" | "), "  (reference: clean 89.7 | 20 dB 87.2 | 10 dB 77.6 | 5 dB 70.7 | 0 dB 56.0)");
