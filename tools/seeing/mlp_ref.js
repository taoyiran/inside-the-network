const H = require("./harness.js"), fs = require("fs");
const train = H.load("train"), test = H.load("test");
for (const hidden of [8, 32]) {
  const net = H.makeMLP(1, hidden), r = H.mulberry32(7), bs = 20;
  for (let epoch = 0; epoch < 6; epoch++) { const lr = epoch < 4 ? 0.1 : 0.03, order = H.shuffle(train, r); for (let s = 0; s + bs <= order.length; s += bs) H.trainStep(net, order.slice(s, s + bs), lr); }
  const params = net.reduce((s, L) => s + (L.W ? L.W.length + L.b.length : 0), 0);
  console.log(`MLP 784-${hidden}-10 (${params} params): test ${(H.accuracy(net, test) * 100).toFixed(1)} train ${(H.accuracy(net, train) * 100).toFixed(1)}`);
  fs.writeFileSync(`${__dirname}/mlp${hidden}_weights.json`, JSON.stringify(net.map((L) => (L.W ? { ...L, W: Array.from(L.W), b: Array.from(L.b) } : L))));
}
