// Loads the model code, the data subset, and a seeded initializer.
const fs = require("fs"), zlib = require("zlib"), path = require("path");
const ctx = {};
new Function("ctx", fs.readFileSync(path.join(__dirname, "cnn.js"), "utf8") + "\nObject.assign(ctx,{grid,forward,backward,softmax,trainStep,convForward,poolForward,reluForward,denseForward});")(ctx);
function load(name) {
  const b = zlib.gunzipSync(fs.readFileSync(path.join(__dirname, "data", name + ".bin.gz"))), n = b.length / 785, out = [];
  for (let i = 0; i < n; i++) { const image = ctx.grid(1, 28, 28); for (let p = 0; p < 784; p++) image.d[p] = b[i * 784 + p] / 255; out.push({ image, label: b[n * 784 + i] }); }
  return out;
}
function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gaussian(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
// He-style init: weights ~ N(0, 2 / fan_in) before ReLU, N(0, 1/fan_in) for the last layer; biases 0
function makeCNN(seed, n1 = 8, n2 = 16) {
  const r = mulberry32(seed), L = (n, c, k) => ({ W: Float64Array.from({ length: n * c * k * k }, () => gaussian(r) * Math.sqrt(2 / (c * k * k))), b: new Float64Array(n) });
  const d = (n, m, s) => ({ type: "dense", n, W: Float64Array.from({ length: n * m }, () => gaussian(r) * Math.sqrt(s / m)), b: new Float64Array(n) });
  return [{ type: "conv", n: n1, k: 5, ...L(n1, 1, 5) }, { type: "relu" }, { type: "pool" }, { type: "conv", n: n2, k: 5, ...L(n2, n1, 5) }, { type: "relu" }, { type: "pool" }, d(10, n2 * 16, 1)];
}
function makeMLP(seed, hidden = 8) {
  const r = mulberry32(seed);
  return [{ type: "dense", n: hidden, W: Float64Array.from({ length: hidden * 784 }, () => gaussian(r) * Math.sqrt(2 / 784)), b: new Float64Array(hidden) }, { type: "relu" }, { type: "dense", n: 10, W: Float64Array.from({ length: 10 * hidden }, () => gaussian(r) * Math.sqrt(1 / hidden)), b: new Float64Array(10) }];
}
const predict = (net, image) => { const s = ctx.forward(net, image).at(-1).d; let b = 0; for (let j = 1; j < 10; j++) if (s[j] > s[b]) b = j; return b; };
const accuracy = (net, set) => set.filter((e) => predict(net, e.image) === e.label).length / set.length;
function shuffle(arr, r) { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
module.exports = { ...ctx, load, mulberry32, gaussian, makeCNN, makeMLP, predict, accuracy, shuffle };
