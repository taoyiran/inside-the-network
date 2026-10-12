const H = require("./harness.js"), P = require("./png.js"), fs = require("fs");
const loadNet = (f) => JSON.parse(fs.readFileSync(`${__dirname}/${f}`, "utf8")).map((L) => (L.W ? { ...L, W: Float64Array.from(L.W), b: Float64Array.from(L.b) } : L));
const cnn = loadNet("ref_weights.json"), test = H.load("test"), r = H.mulberry32(11);
const probs = (x) => H.softmax(H.forward(cnn, x).at(-1).d);
for (const [name, gen] of [["uniform noise", () => r()], ["sparse noise (10% ink)", () => (r() < 0.1 ? 1 : 0)], ["blank", () => 0], ["all ink", () => 1]]) {
  const confs = [], cls = new Array(10).fill(0);
  for (let t = 0; t < (name === "blank" || name === "all ink" ? 1 : 200); t++) { const x = H.grid(1, 28, 28); for (let p = 0; p < 784; p++) x.d[p] = gen(); const p = probs(x), k = p.indexOf(Math.max(...p)); confs.push(p[k]); cls[k]++; }
  confs.sort((a, b) => a - b);
  console.log(`${name}: median top confidence ${(100 * confs[Math.floor(confs.length / 2)]).toFixed(0)}%, share ≥ 90%: ${(100 * confs.filter((c) => c >= 0.9).length / confs.length).toFixed(0)}%, answers ${cls.join(",")}`);
}
// test set: how often is the network ≥ 99% sure, and how accurate then
const ps = test.map((e) => ({ p: probs(e.image), y: e.label }));
const sure = ps.filter((o) => Math.max(...o.p) >= 0.99);
console.log(`test digits ≥99% sure: ${(100 * sure.length / ps.length).toFixed(0)}%, accuracy among them ${(100 * sure.filter((o) => o.p.indexOf(Math.max(...o.p)) === o.y).length / sure.length).toFixed(1)}%`);
// big render of final filters; per-filter max |w|
P.tilesPNG(`${__dirname}/filters_final.png`, [Array.from({ length: 8 }, (_, f) => ({ data: cnn[0].W.slice(f * 25, f * 25 + 25), w: 5, h: 5, kind: "div" }))], 24, 8);
console.log("conv1 biases", Array.from(cnn[0].b, (v) => v.toFixed(2)).join(" "));
// which conv1 filters are ever active on test digits?
const active = new Array(8).fill(0); for (const e of test.slice(0, 200)) { const v = H.forward(cnn, e.image)[2]; for (let f = 0; f < 8; f++) for (let i = 0; i < 576; i++) if (v.d[f * 576 + i] > 0) { active[f]++; break; } }
console.log("conv1 filters active on how many of 200 digits:", active.join(" "));
