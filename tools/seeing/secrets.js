const H = require("./harness.js"), P = require("./png.js"), fs = require("fs");
const test = H.load("test");
const loadNet = (f) => JSON.parse(fs.readFileSync(`${__dirname}/${f}`, "utf8")).map((L) => (L.W ? { ...L, W: Float64Array.from(L.W), b: Float64Array.from(L.b) } : L));
const cnn = loadNet("ref_weights.json"), mlp8 = loadNet("mlp8_weights.json"), mlp32 = loadNet("mlp32_weights.json");
const img = (d) => { const g = H.grid(1, 28, 28); g.d.set(d); return g; };
// gradient of a chosen objective w.r.t. pixels: backward() uses label for p − y; for "loss of label" this is ∂(−log p_label)/∂x
const gradLoss = (net, x, label) => H.backward(net, H.forward(net, x), label, net.map((L) => (L.W ? { W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) } : null)));
const probs = (net, x) => H.softmax(H.forward(net, x).at(-1).d);
// 1. adversarial: untargeted iterative sign steps, L∞ budget eps
function attack(net, e, eps, steps = 10) {
  const x = img(e.image.d), a = eps / 4;
  for (let s = 0; s < steps; s++) { const g = gradLoss(net, x, e.label); for (let p = 0; p < 784; p++) { const v = x.d[p] + a * Math.sign(g[p]); x.d[p] = Math.max(0, Math.min(1, Math.max(e.image.d[p] - eps, Math.min(e.image.d[p] + eps, v)))); } }
  return x;
}
const right = test.filter((e) => H.predict(cnn, e.image) === e.label).slice(0, 300), r = H.mulberry32(3);
for (const eps of [0.05, 0.1, 0.15, 0.2, 0.25]) {
  let fooled = 0, randomFooled = 0, fgsm = 0, conf = 0;
  for (const e of right) {
    const x = attack(cnn, e, eps), p = probs(cnn, x), k = p.indexOf(Math.max(...p)); if (k !== e.label) { fooled++; conf += p[k]; }
    const g = gradLoss(cnn, e.image, e.label), x1 = img(e.image.d.map((v, i) => Math.max(0, Math.min(1, v + eps * Math.sign(g[i]))))); if (H.predict(cnn, x1) !== e.label) fgsm++;
    const xr = img(e.image.d.map((v) => Math.max(0, Math.min(1, v + eps * (r() < 0.5 ? -1 : 1))))); if (H.predict(cnn, xr) !== e.label) randomFooled++;
  }
  console.log(`eps ${eps}: 10-step attack fools ${(100 * fooled / right.length).toFixed(0)}% (mean confidence in wrong answer ${(100 * conf / Math.max(1, fooled)).toFixed(0)}%), one-step ${(100 * fgsm / right.length).toFixed(0)}%, random ±eps ${(100 * randomFooled / right.length).toFixed(0)}%`);
}
// pictures: a few originals, attacked (eps 0.15), difference
{
  const rows = [[], [], []];
  for (const e of right.slice(0, 8)) { const x = attack(cnn, e, 0.15), p = probs(cnn, x); rows[0].push({ data: e.image.d, w: 28, h: 28, kind: "gray" }); rows[1].push({ data: x.d, w: 28, h: 28, kind: "gray" }); rows[2].push({ data: x.d.map((v, i) => v - e.image.d[i]), w: 28, h: 28, kind: "div" }); console.log(`  example: label ${e.label} → attacked prediction ${p.indexOf(Math.max(...p))} at ${(100 * Math.max(...p)).toFixed(0)}%`); }
  P.tilesPNG(`${__dirname}/adv.png`, rows, 4, 6);
}
// 2. shifts
function shift(d, dx, dy) { const out = new Float64Array(784); for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) { const sx = x - dx, sy = y - dy; if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) out[y * 28 + x] = d[sy * 28 + sx]; } return out; }
for (const k of [0, 1, 2, 3, 4, 5]) {
  const acc = (net) => { let ok = 0, n = 0; for (const e of test) for (const [dx, dy] of [[k, 0], [-k, 0], [0, k], [0, -k]]) { n++; if (H.predict(net, img(shift(e.image.d, dx, dy))) === e.label) ok++; } return (100 * ok / n).toFixed(1); };
  console.log(`shift ${k}px: CNN ${acc(cnn)}%  MLP-8 ${acc(mlp8)}%  MLP-32 ${acc(mlp32)}%`);
}
// 3. dreams: gradient ascent on the class score from a blank image, pixels kept in [0, 1], small L2 pull to 0
function dream(net, cls, steps = 300, lr = 0.05, l2 = 0.02) {
  const x = img(new Float64Array(784).fill(0.0));
  for (let s = 0; s < steps; s++) {
    const values = H.forward(net, x), grads = net.map((L) => (L.W ? { W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) } : null));
    // backward() starts from p − y; to push the raw score of cls up we use the loss of cls (raises p_cls): gradient descent on −log p_cls
    const g = H.backward(net, values, cls, grads);
    for (let p = 0; p < 784; p++) x.d[p] = Math.max(0, Math.min(1, x.d[p] - lr * (g[p] + l2 * x.d[p])));
  }
  return x;
}
const dreams = Array.from({ length: 10 }, (_, c) => dream(cnn, c));
console.log("dream confidences:", dreams.map((x, c) => (100 * probs(cnn, x)[c]).toFixed(0)).join(" "));
P.tilesPNG(`${__dirname}/dreams.png`, [dreams.map((x) => ({ data: x.d, w: 28, h: 28, kind: "gray" }))], 4, 6);
// 4. filters, feature maps of one digit, saliency
const e0 = test[0], vals = H.forward(cnn, e0.image);
P.tilesPNG(`${__dirname}/fmaps.png`, [
  [{ data: e0.image.d, w: 28, h: 28, kind: "gray" }],
  Array.from({ length: 8 }, (_, f) => ({ data: cnn[0].W.slice(f * 25, f * 25 + 25), w: 5, h: 5, kind: "div" })),
  Array.from({ length: 8 }, (_, f) => ({ data: vals[2].d.slice(f * 576, f * 576 + 576), w: 24, h: 24, kind: "div" })),
  Array.from({ length: 16 }, (_, f) => ({ data: vals[5].d.slice(f * 64, f * 64 + 64), w: 8, h: 8, kind: "div" })),
], 4, 6);
console.log("digit", e0.label, "probs", probs(cnn, e0.image).map((v) => v.toFixed(2)).join(" "));
const sal = right.slice(0, 8).map((e) => { const g = gradLoss(cnn, e.image, e.label); return { data: g.map((v) => -v), w: 28, h: 28, kind: "div" }; });
P.tilesPNG(`${__dirname}/saliency.png`, [right.slice(0, 8).map((e) => ({ data: e.image.d, w: 28, h: 28, kind: "gray" })), sal], 4, 6);
