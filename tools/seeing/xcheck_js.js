const H = require("./harness.js"), fs = require("fs");
const train = H.load("train");
const net = H.makeCNN(3); net[0].W.forEach((v, i) => { net[0].W[i] = v * 0.02; }); net[0].b.fill(0.1);
const batch = train.slice(0, 6);
fs.writeFileSync(__dirname + "/xcheck_in.json", JSON.stringify({ net: net.map((L) => (L.W ? { ...L, W: Array.from(L.W), b: Array.from(L.b) } : L)), batch: batch.map((e) => [Array.from(e.image.d), e.label]) }));
const out = []; for (let s = 0; s < 3; s++) out.push(H.trainStep(net, s % 2 ? batch.slice(3) : batch.slice(0, 3), 0.1));
fs.writeFileSync(__dirname + "/xcheck_js.json", JSON.stringify({ out, net: net.map((L) => (L.W ? { W: Array.from(L.W), b: Array.from(L.b) } : null)) }));
console.log("JS:", out.map((o) => o.loss.toFixed(12)).join(" "));
