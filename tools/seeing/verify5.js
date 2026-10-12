const P = require("./pagecode5.js"), H = require("./harness.js"), fs = require("fs");
const ref = JSON.parse(fs.readFileSync(__dirname + "/ref_weights.json", "utf8"));
// 1. weights file decodes to the trained weights exactly
let same = true; [0, 3, 6].forEach((l) => { ref[l].W.forEach((v, i) => { if (P.NET[l].W[i] !== v) same = false; }); ref[l].b.forEach((v, i) => { if (P.NET[l].b[i] !== v) same = false; }); });
console.log("weights.js == trained weights:", same);
// 2. page init == the init used to train the reference
const a = P.makeCNN(1), b = H.makeCNN(1); b[0].W.forEach((v, i) => { b[0].W[i] = v * 0.02; }); b[0].b.fill(0.1);
console.log("page makeCNN(1) == training init:", [0, 3, 6].every((l) => a[l].W.every((v, i) => v === b[l].W[i]) && a[l].b.every((v, i) => v === b[l].b[i])));
// 3. page trainer(1) for 6 passes reproduces the reference exactly (node)
const T = P.trainer(1); while (T.step < 1800) T.next();
console.log("page trainer, 1,800 steps == reference:", [0, 3, 6].every((l) => T.net[l].W.every((v, i) => v === ref[l].W[i])));
// 4. accuracies and shifts
const test = P.DATA.test, acc = (net, f) => test.filter((e) => P.argmax(P.forward(net, f(e.image)).at(-1).d) === e.label).length / test.length;
console.log("centered: plain", (100 * acc(P.PLAIN, (x) => x)).toFixed(1), "conv", (100 * acc(P.NET, (x) => x)).toFixed(1));
const avg = (net, k) => [[k, 0], [-k, 0], [0, k], [0, -k]].reduce((s, [dx, dy]) => s + acc(net, (x) => P.shifted(x, dx, dy)), 0) / 4;
console.log("shift 2 px, average of 4 directions: plain", (100 * avg(P.PLAIN, 2)).toFixed(1), "conv", (100 * avg(P.NET, 2)).toFixed(1));
// 5. the attack, with the page's own fool(), on the first 300 test digits it reads correctly
const right = test.filter((e) => P.argmax(P.probsOf(P.NET, e.image)) === e.label).slice(0, 300), r = P.mulberry32(3);
for (const eps of [0.05, 0.1, 0.15, 0.2]) {
  let fooled = 0, rnd = 0, conf = 0;
  for (const e of right) {
    const x = P.fool(e.image, e.label, eps, -1).at(-1), p = P.probsOf(P.NET, Object.assign(P.grid(1, 28, 28), { d: x })), k = P.argmax(p); if (k !== e.label) { fooled++; conf += p[k]; }
    const xr = e.image.d.map((v) => Math.max(0, Math.min(1, v + eps * (r() < 0.5 ? -1 : 1)))); if (P.argmax(P.probsOf(P.NET, Object.assign(P.grid(1, 28, 28), { d: xr }))) !== e.label) rnd++;
  }
  console.log(`eps ${eps}: fooled ${(100 * fooled / 300).toFixed(1)}% (mean confidence ${(100 * conf / Math.max(1, fooled)).toFixed(0)}%), random ±eps ${(100 * rnd / 300).toFixed(1)}%`);
}
// 6. dreams: confidence for each digit; average-digit confidence
const dreams = Array.from({ length: 10 }, (_, d) => { const D = P.dreamer(d); while (D.step < 300) D.next(); return P.probsOf(P.NET, D.x)[d]; });
const avgs = Array.from({ length: 10 }, (_, d) => P.probsOf(P.NET, Object.assign(P.grid(1, 28, 28), { d: P.averageOf(d) }))[d]);
console.log("dream confidence:", dreams.map((v) => (100 * v).toFixed(1)).join(" "));
console.log("average-digit confidence:", avgs.map((v) => (100 * v).toFixed(1)).join(" "));
// 7. parameters
const count = (net) => net.reduce((s, L) => s + (L.W ? L.W.length + L.b.length : 0), 0);
console.log("parameters: conv", count(P.NET), "plain", count(P.PLAIN), "| layers", [0, 3, 6].map((l) => P.NET[l].W.length + P.NET[l].b.length).join(" + "));
