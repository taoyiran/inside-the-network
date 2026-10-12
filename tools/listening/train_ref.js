// The reference networks for 5′: chapter 5's code, centered 32 × 32 spectrograms, 800 clips per digit.
const A = require("./audioharness.js"), H = require("../seeing/harness.js"), fs = require("fs"), path = require("path");
function load(set) { const b = fs.readFileSync(path.join(__dirname, "data", set + ".bin")), N = b.length / 1025, out = []; for (let i = 0; i < N; i++) { const g = A.grid(1, 32, 32); for (let q = 0; q < 1024; q++) g.d[q] = Math.round((b[i * 1024 + q] / 255) * 15) / 15 - 0.5; out.push({ image: g, label: b[N * 1024 + i] }); } return out; }
const train = load("train"), test = load("test");
function makeCNN(seed) { const r = H.mulberry32(seed), conv = (n, c) => ({ type: "conv", n, k: 5, W: Float64Array.from({ length: n * c * 25 }, () => H.gaussian(r) * Math.sqrt(2 / (c * 25))), b: new Float64Array(n) });
  const net = [conv(8, 1), { type: "relu" }, { type: "pool" }, conv(16, 8), { type: "relu" }, { type: "pool" }, { type: "dense", n: 10, W: Float64Array.from({ length: 4000 }, () => H.gaussian(r) * Math.sqrt(1 / 400)), b: new Float64Array(10) }];
  if (process.env.SMALL !== "0") { net[0].W.forEach((v, i) => { net[0].W[i] = v * 0.02; }); net[0].b.fill(0.1); } return net; }
function makeMLP(seed) { const r = H.mulberry32(seed); return [{ type: "dense", n: 8, W: Float64Array.from({ length: 8192 }, () => H.gaussian(r) * Math.sqrt(2 / 1024)), b: new Float64Array(8) }, { type: "relu" }, { type: "dense", n: 10, W: Float64Array.from({ length: 80 }, () => H.gaussian(r) * Math.sqrt(1 / 8)), b: new Float64Array(10) }]; }
const acc = (net, set) => set.filter((e) => { const s = A.forward(net, e.image).at(-1).d; let b = 0; for (let j = 1; j < 10; j++) if (s[j] > s[b]) b = j; return b === e.label; }).length / set.length;
function run(net, lr0, passes = 10) { const r = H.mulberry32(7); for (let p = 1; p <= passes; p++) { const order = H.shuffle(train, r), lr = p <= passes - 2 ? lr0 : lr0 * 0.3; for (let s = 0; s + 20 <= order.length; s += 20) A.trainStep(net, order.slice(s, s + 20), lr); } return net; }
const ser = (net) => net.map((L) => (L.W ? { ...L, W: Array.from(L.W), b: Array.from(L.b) } : L));
const which = process.argv[2];
if (which === "cnn") { const net = run(makeCNN(1), 0.1); console.log("CNN test", (100 * acc(net, test)).toFixed(1), "train", (100 * acc(net, train)).toFixed(1)); fs.writeFileSync(path.join(__dirname, process.env.SMALL === "0" ? "ref_cnn_normalinit.json" : "ref_cnn.json"), JSON.stringify(ser(net))); }
else for (const lr of [0.1, 0.03]) { const net = run(makeMLP(1), lr); console.log("plain lr", lr, "test", (100 * acc(net, test)).toFixed(1), "train", (100 * acc(net, train)).toFixed(1)); fs.writeFileSync(path.join(__dirname, `ref_mlp_${lr}.json`), JSON.stringify(ser(net))); }
