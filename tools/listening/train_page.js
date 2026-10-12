// Train one of chapter 5′'s networks with the page's own trainer (4,000 steps, as on the page) and save it:
// node train_page.js cnn | plain | shifted   (shifted: the convolutional network trained on pictures moved up or
// down by up to 3 bands; it is saved as final_aug.json). verify5p.js checks the shipped weights against these.
const P = require("./pageinit_node.js"), fs = require("fs"), path = require("path");
const which = process.argv[2], T = which === "shifted" ? P.trainer(1, P.makeCNN, 3) : P.trainer(1, which === "cnn" ? P.makeCNN : P.makePlain), t0 = Date.now(), log = [];
while (T.step < 4000) { T.next(); if (T.step % 400 === 0) log.push((100 * P.accuracy(T.net)).toFixed(1)); }
console.log(which, "test per pass:", log.join(" "), "| train", (100 * P.accuracy(T.net, P.DATA.train)).toFixed(1), "|", ((Date.now() - t0) / 1000).toFixed(0), "s");
fs.writeFileSync(path.join(__dirname, `final_${which === "shifted" ? "aug" : which}.json`), JSON.stringify(T.net.map((L) => (L.W ? { ...L, W: Array.from(L.W), b: Array.from(L.b) } : L))));
