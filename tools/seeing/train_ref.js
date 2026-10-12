// Train the reference network with the page's code (first-layer weights start tiny), save the weights.
const H = require("./harness.js"), P = require("./png.js"), fs = require("fs");
const train = H.load("train"), test = H.load("test");
const net = H.makeCNN(1); net[0].W.forEach((v, i) => { net[0].W[i] = v * 0.02; }); net[0].b.fill(0.1);
const r = H.mulberry32(7), bs = 20; const log = []; const t0 = Date.now();
for (let epoch = 0; epoch < 6; epoch++) {
  const lr = epoch < 4 ? 0.1 : 0.03, order = H.shuffle(train, r);
  for (let s = 0; s + bs <= order.length; s += bs) H.trainStep(net, order.slice(s, s + bs), lr);
  log.push(`ep${epoch + 1}:${(H.accuracy(net, test) * 100).toFixed(1)}/${(H.accuracy(net, train) * 100).toFixed(1)}`);
}
console.log("test/train per epoch:", log.join(" "), "|", ((Date.now() - t0) / 1000).toFixed(1), "s");
fs.writeFileSync(__dirname + "/ref_weights.json", JSON.stringify(net.map((L) => (L.W ? { ...L, W: Array.from(L.W), b: Array.from(L.b) } : L))));
