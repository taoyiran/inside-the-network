// Causal checks for Secret 3: (a) train with random pitch shifts (±3 bands): does pitch robustness appear?
// (b) train on time-aligned spectrograms (energy centroid moved to column 16): does time robustness disappear?
const P = require("../pagecode5p.js"), D = P.DATA;
const mode = process.argv[2];
const centroid = (s) => { const col = Array.from({ length: 32 }, (_, c) => { let m = 0; for (let r = 0; r < 32; r++) m += s.d[r * 32 + c]; return m; }); const mn = Math.min(...col); let t = 0, w = 0; col.forEach((v, c) => { t += (v - mn) * c; w += v - mn; }); return t / w; };
const align = (s) => P.moved(s, Math.round(16 - centroid(s)), 0);
function train(transform, seed = 1) {
  const net = P.makeCNN(seed), r = P.mulberry32(6 + seed), ra = P.mulberry32(99);
  let order = null, pos = 0, pass = 0;
  for (let step = 0; step < 4000; step++) {
    if (!order || pos >= order.length) { order = Array.from({ length: D.train.n }, (_, i) => i); for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } pos = 0; pass++; }
    const batch = order.slice(pos, pos + 20).map((k) => ({ image: P.toInput(transform(P.specOf(D.train, k), ra)), label: D.train.labels[k] }));
    pos += 20; P.trainStep(net, batch, pass <= 8 ? 0.1 : 0.03);
  }
  return net;
}
const acc = (net, f) => { let c = 0; for (let k = 0; k < D.test.n; k++) if (P.argmax(P.probsOf(net, f(P.specOf(D.test, k)))) === D.test.labels[k]) c++; return (c / 10).toFixed(1); };
if (mode === "pitch") {
  const net = train((s, r) => P.moved(s, 0, Math.floor(r() * 7) - 3));
  console.log("pitch-augmented cnn: unmoved", acc(net, (s) => s), "| up 1/2/3:", [1, 2, 3].map((d) => acc(net, (s) => P.moved(s, 0, d))).join(" "), "| down 2:", acc(net, (s) => P.moved(s, 0, -2)), "| later 2/4/6:", [2, 4, 6].map((d) => acc(net, (s) => P.moved(s, d, 0))).join(" "));
  console.log("reference cnn:        unmoved", acc(P.NET, (s) => s), "| up 1/2/3:", [1, 2, 3].map((d) => acc(P.NET, (s) => P.moved(s, 0, d))).join(" "));
} else if (mode === "align") {
  const net = train((s) => align(s));
  console.log("time-aligned training, on aligned test: unmoved", acc(net, (s) => align(s)), "| later 2/4/6:", [2, 4, 6].map((d) => acc(net, (s) => P.moved(align(s), d, 0))).join(" "), "| earlier 2/4:", [-2, -4].map((d) => acc(net, (s) => P.moved(align(s), d, 0))).join(" "), "| up 2/3:", [2, 3].map((d) => acc(net, (s) => P.moved(align(s), 0, d))).join(" "), "| raw test:", acc(net, (s) => s));
  console.log("reference cnn on aligned test: unmoved", acc(P.NET, (s) => align(s)), "| later 2/4/6:", [2, 4, 6].map((d) => acc(P.NET, (s) => P.moved(align(s), d, 0))).join(" "));
}
