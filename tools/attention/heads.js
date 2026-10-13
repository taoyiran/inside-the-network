// What each head does, over the page's 100 test stories (as the page and package.js measure it): where its weight goes by distance back, the
// tokens it reads most (by total weight, against how common they are), and its distance scores R.
// node heads.js A_20000.bin
const L = require("./lib.js");
const m = L.loadModel(process.argv[2]), tk = L.tokenizer(), V = tk.vocab;
const vs = L.storiesOf("test.bin").stories.slice(0, 100);
const bins = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 7], [8, 15], [16, 63], [64, 255]];
const byDist = Array.from({ length: m.H }, () => new Float64Array(bins.length)), byTok = Array.from({ length: m.H }, () => new Float64Array(m.V)), freq = new Float64Array(m.V);
let N = 0, ent = new Float64Array(m.H);
for (const { tokens } of vs) {
  const f = L.ATN.forward(m, tokens);
  for (let i = 0; i < tokens.length; i++) {
    N++; freq[tokens[i]]++;
    for (let h = 0; h < m.H; h++) for (let t = 0; t <= Math.min(i, m.T - 1); t++) {
      const w = f.w[(i * m.H + h) * m.T + t];
      byTok[h][tokens[i - t]] += w; if (w > 0) ent[h] -= w * Math.log(w);
      for (let b = 0; b < bins.length; b++) if (t >= bins[b][0] && t <= bins[b][1]) byDist[h][b] += w;
    }
  }
}
console.log("distance bins:", bins.map(([a, b]) => (a === b ? a : `${a}-${b}`)).join(" "));
for (let h = 0; h < m.H; h++) {
  const top = Array.from(byTok[h].keys()).filter((x) => freq[x] >= 5).sort((a, b) => byTok[h][b] / freq[b] - byTok[h][a] / freq[a]).slice(0, 12);
  console.log(`head ${h + 1}: ${Array.from(byDist[h], (x) => ((100 * x) / N).toFixed(0).padStart(3)).join(" ")} | spread ${(ent[h] / N).toFixed(2)} nats | R ${Array.from(m.R.subarray(h * m.T, h * m.T + 6), (x) => x.toFixed(1)).join(" ")} … R[30] ${m.R[h * m.T + 30].toFixed(1)}`);
  console.log(`   reads most, per appearance: ${top.map((x) => `${V[x]} ${(byTok[h][x] / freq[x]).toFixed(1)}`).join(", ")}`);
}
