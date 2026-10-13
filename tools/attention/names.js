// Names, far back. In the test stories, at every position whose right answer is the story's name (a one-token name
// such as "lily") and whose last mention is more than 8 tokens back:
//   1. how often each method's first guess is the name (counting with the last 4 tokens, chapter 6, attention);
//   2. swap: the name replaced everywhere by another one-token name that is not in the story; how often the
//      attention network's first guess is now the new name;
//   3. made up: the name replaced by an invented name of 2 or 3 pieces ("z ##ark ##o"); after the first piece of a
//      later mention, how often the first guess is the next piece (finishing a name seen earlier in the story: an
//      induction pattern), against the same at the first mention, where only the training stories can help.
// node names.js A_20000.bin   (NSTORIES=1000 by default: the test stories)
const L = require("./lib.js");
const NS = +(process.env.NSTORIES || 1000), K = 8;
const tk = L.tokenizer(), V = tk.vocab;
const NAMES = L.names();
const tok = L.loadTokens("train.bin"), te = L.loadTokens("test.bin");
// invented names of 2 or 3 pieces
const INVENT = "vorna quilla dashu morvin tazzle sindra kavu wemble jorby zarko tilda yurin zibble".split(" ")
  .map((w) => tk.encode(w)).filter((ids) => ids.length >= 2 && ids.length <= 3);
const { stories } = L.storiesOf("test.bin");
const ch6 = L.W6.loadModel(process.env.CH6 || "Q128_600000.bin");
const nets = process.argv.slice(2).filter((x) => !x.includes("=")).map((f) => [f, L.loadModel(f)]);
const argmax = L.argmax;
// counting with the last 4 tokens, falling back to fewer, over the training stories
const key = (c, l) => c.slice(K - l).join(","), want = Array.from({ length: 5 }, () => new Map());
const ctxOf = (toks, r) => Array.from({ length: K }, (_, j) => { const i = r - K + 1 + j; return i >= 1 ? toks[i] : 0; }); // the 8 tokens up to position r, story only
const st = L.starts(tok);
console.log("names:", NAMES.map((x) => V[x]).join(" "), "| invented:", INVENT.map((x) => x.map((y) => V[y]).join(" ")).join(", "));
const cases = [];
stories.slice(0, NS).forEach(({ tokens, next }, s) => {
  const X = L.nameOf(tokens);
  if (X < 0) return;
  const cnt = new Map(); for (const x of tokens) if (NAMES.includes(x)) cnt.set(x, (cnt.get(x) || 0) + 1);
  const Y = NAMES.find((y, i) => !cnt.has(y) && i >= (s % NAMES.length)) ?? NAMES.find((y) => !cnt.has(y));
  const Z = INVENT[s % INVENT.length];
  const at = L.farPositions(tokens, next, X); // positions r (predicting next[r]) where the answer is X and X's last mention is more than 8 back
  if (at.length) cases.push({ s, tokens, next, X, Y, Z, at });
});
for (const { tokens, at } of cases) for (const r of at) { const c = ctxOf(tokens, r); for (let l = 1; l <= 4; l++) if (!want[l].has(key(c, l))) want[l].set(key(c, l), new Map()); }
for (let t = 0; t < tok.length; t++) { const c = L.ctxAt(tok, st, t, K); for (let l = 1; l <= 4; l++) { const m = want[l].get(key(c, l)); if (!m) break; m.set(tok[t], (m.get(tok[t]) || 0) + 1); } }
const best = (m) => { let b = -1, bn = 0; for (const [x, v] of m) if (v > bn || (v === bn && x < b)) { bn = v; b = x; } return b; };
const count4 = (c) => { let g = -1; for (let l = 1; l <= 4; l++) { const m = want[l].get(key(c, l)); if (m && m.size) g = best(m); else break; } return g; };
const N = cases.reduce((a, c) => a + c.at.length, 0);
let c4 = 0, c6 = 0;
for (const { tokens, X, at } of cases) for (const r of at) { const c = ctxOf(tokens, r); if (count4(c) === X) c4++; const p = L.NWM.forward(ch6, c).p; if (argmax(p, 0, p.length) === X) c6++; }
console.log(`${cases.length} stories, ${N} positions whose answer is the name, last mentioned more than 8 tokens back`);
console.log(`counting (last 4): ${((100 * c4) / N).toFixed(1)}%   chapter 6: ${((100 * c6) / N).toFixed(1)}%`);
for (const [f, m] of nets) {
  let hit = 0, pr = 0, swapHit = 0, swapOld = 0, z1 = 0, z2 = 0, z2n = 0, z1n = 0;
  for (const { tokens, next, X, Y, Z, at } of cases) {
    const f0 = L.ATN.forward(m, tokens);
    for (const r of at) { if (argmax(f0.p, r * m.V, m.V) === X) hit++; pr += f0.p[r * m.V + X]; }
    const sw = Array.from(tokens, (x) => (x === X ? Y : x)), f1 = L.ATN.forward(m, sw);
    for (const r of at) { const g = argmax(f1.p, r * m.V, m.V); if (g === Y) swapHit++; if (g === X) swapOld++; }
    // the invented name: every X becomes the two pieces Z; check positions that were in `at`
    const zt = [], map = []; for (let i = 0; i < tokens.length; i++) { map.push(zt.length); if (tokens[i] === X) zt.push(...Z); else zt.push(tokens[i]); }
    const nextZ = [...zt.slice(1), next[next.length - 1]];
    const f2 = L.ATN.forward(m, zt);
    const first = map[tokens.indexOf(X)] - 1; // the position just before the first mention's first piece
    for (let j = 1; j < Z.length; j++) { z1n++; if (argmax(f2.p, (first + j) * m.V, m.V) === Z[j]) z1++; }
    for (const r of at) {
      const r1 = map[r]; // the token after position r1 is the first piece of a later mention
      if (nextZ[r1] !== Z[0]) throw new Error("misaligned");
      for (let j = 1; j < Z.length; j++) { z2n++; if (argmax(f2.p, (r1 + j) * m.V, m.V) === Z[j]) z2++; }
    }
  }
  console.log(`${f}: first guess is the name ${((100 * hit) / N).toFixed(1)}% (mean probability ${(pr / N).toFixed(3)}) | swapped: new name ${((100 * swapHit) / N).toFixed(1)}%, old name ${((100 * swapOld) / N).toFixed(1)}% | invented, the next piece of the name: at the first mention ${((100 * z1) / z1n).toFixed(1)}%, at later mentions ${((100 * z2) / z2n).toFixed(1)}%`);
}
