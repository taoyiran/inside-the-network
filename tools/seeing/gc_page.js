const H = require("./harness.js"), fs = require("fs");
const net = JSON.parse(fs.readFileSync(__dirname + "/ref_weights.json", "utf8")).map((L) => (L.W ? { ...L, W: Float64Array.from(L.W), b: Float64Array.from(L.b) } : L));
const test = H.load("test"), r = H.mulberry32(5);
net.forEach((L) => L.b && L.b.forEach((_, i) => { L.b[i] += H.gaussian(r) * 0.1; }));
const e = test[0], x = H.grid(1, 28, 28); e.image.d.forEach((v, i) => { x.d[i] = v + H.gaussian(r) * 0.05; });
const grads = net.map((L) => (L.W ? { W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) } : null));
const LBL = (e.label + 3) % 10; const gx = H.backward(net, H.forward(net, x), LBL, grads);
const loss = () => -Math.log(H.softmax(H.forward(net, x).at(-1).d)[LBL]);
const num = (arr, i, h) => { const v = arr[i]; arr[i] = v + h; const a = loss(); arr[i] = v - h; const b = loss(); arr[i] = v; return (a - b) / (2 * h); };
const rows = [];
const check = (name, arr, i, g) => { const n6 = num(arr, i, 1e-5), n4 = num(arr, i, 1e-4), n8 = num(arr, i, 1e-8); rows.push({ name, i, g, n6, rel: Math.abs(n6 - g) / Math.max(1e-6, Math.abs(n6) + Math.abs(g)), n4, n8 }); };
// only gradients big enough to measure: |g| ≥ 1e-3
const pick = (arr, gs, name, want) => { for (let t = 0, got = 0; t < 5000 && got < want; t++) { const i = Math.floor(r() * arr.length); if (Math.abs(gs[i]) >= 1e-3) { check(name, arr, i, gs[i]); got++; } } };
for (const l of [0, 3, 6]) pick(net[l].W, grads[l].W, "W" + l, 10);
pick(x.d, gx, "px", 20);
console.log("checks", rows.length);
rows.sort((a, b) => b.rel - a.rel);
for (const o of rows.slice(0, 5)) console.log(o.name, o.i, "backprop", o.g.toExponential(4), "numeric h=1e-6", o.n6.toExponential(4), "h=1e-4", o.n4.toExponential(4), "h=1e-8", o.n8.toExponential(4), "rel", o.rel.toExponential(1));
// min |z| over the pre-activation values, and the loss itself
const v = H.forward(net, x); let mz = Infinity; for (const k of [1, 4]) for (const z of v[k].d) mz = Math.min(mz, Math.abs(z));
console.log("loss", loss().toExponential(3), "| smallest |z| before a ReLU", mz.toExponential(2));
