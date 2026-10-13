// Secret 2, per new name: in the page's test stories whose name comes back from further than 8 tokens, the name is
// replaced everywhere by each new name in turn (stories that already contain it are skipped), and at the places where
// the name comes back, the network's first guesses are counted. Run it on several checkpoints to see how much the
// per-name rates move. node swapnames.js F8_19000.bin   (NS=100 stories; W="lily tom …" to choose the names)
const L = require("./lib.js");
const tk = L.tokenizer(), V = tk.vocab, m = L.loadModel(process.argv[2]);
const cases = [];
L.storiesOf("test.bin").stories.slice(0, +(process.env.NS || 100)).forEach(({ tokens, next }) => {
  const X = L.nameOf(tokens); if (X < 0) return;
  const at = L.farPositions(tokens, next, X); if (at.length) cases.push({ tokens, X, at });
});
for (const w of (process.env.W || "lily tom max mia sue ben fred").split(" ")) {
  const Y = tk.id.get(w), got = new Map(); let n = 0;
  for (const { tokens, X, at } of cases) {
    if (tokens.includes(Y)) continue;
    const f = L.ATN.forward(m, Uint16Array.from(tokens, (x) => (x === X ? Y : x)));
    for (const r of at) { n++; const b = V[L.argmax(f.p, r * m.V, m.V)]; got.set(b, (got.get(b) || 0) + 1); }
  }
  console.log(w.padEnd(5), String(n).padStart(4), "places |", [...got].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k, v]) => `${JSON.stringify(k)} ${((100 * v) / n).toFixed(1)}%`).join(", "));
}
