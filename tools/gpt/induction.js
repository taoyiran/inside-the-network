// Induction, measured two ways, for any checkpoints (of chapter 8's GPT, or chapter 7's network with ch7=1):
//   1. invented names, as chapter 7's Secret 4: the hero of each of the page's name stories (test stories among the
//      first 100 whose name comes back from further than 8 tokens) is given each of 6 invented names in turn; at
//      every mention, each piece after the first: is it the first guess? Split by first and later mentions, and by
//      whether that pair of pieces ever appears in the training stories.
//   2. repeated random tokens: <end>, then 40 tokens drawn at random from the 1,000 most common, then the same 40
//      again (the page's 50 sequences); how often the next token is the first guess in the second copy (where it can
//      be copied from the first), against the first copy (where it is random). The guess of the second copy's first
//      token, which nothing before can tell, counts in neither.
// node induction.js G_10000.bin [G_20000.bin …]
const L = require("./lib.js");
const tk = L.tokenizer(), V = tk.vocab, tok = L.loadTokens("train.bin");
const INVENT = L.INVENT;
const test = L.storiesOf("test.bin").stories.slice(0, 100), list = [];
test.forEach(({ tokens, next }) => { const X = L.hero(tokens); if (X >= 0 && L.farPositions(tokens, next, X).length) list.push({ tokens, X }); });
const pairs = new Map();
for (const w of INVENT) { const Z = tk.encode(w); for (let j = 1; j < Z.length; j++) pairs.set(Z[j - 1] * 2000 + Z[j], 0); }
for (let t = 0; t + 1 < tok.length; t++) { const k = tok[t] * 2000 + tok[t + 1]; if (pairs.has(k)) pairs.set(k, pairs.get(k) + 1); }
const reps = L.pageSetup().reps; // the same sequences as the page's
function measure(fwd, VV) {
  const c = { seen: [0, 0, 0, 0], unseen: [0, 0, 0, 0] }; // [first right, first n, later right, later n]
  for (const w of INVENT) {
    const Z = tk.encode(w);
    for (const { tokens, X } of list) {
      const { toks: out, ment } = L.invented(tokens, X, Z); // the name replaced where it stands as a word, as on the page
      const p = fwd(out);
      ment.forEach((mm, k) => { for (let j = 1; j < Z.length; j++) { const ok = L.argmax(p, (mm + j - 1) * VV, VV) === Z[j], h = c[pairs.get(Z[j - 1] * 2000 + Z[j]) ? "seen" : "unseen"]; h[k ? 2 : 0] += ok; h[k ? 3 : 1]++; } });
    }
  }
  let r1 = 0, r2 = 0, n1 = 0, n2 = 0, l2 = 0;
  for (const s of reps) {
    const p = fwd(s);
    for (let i = 1; i < s.length - 1; i++) { if (i === 40) continue; const ok = L.argmax(p, i * VV, VV) === s[i + 1]; if (i < 40) { r1 += ok; n1++; } else { r2 += ok; n2++; l2 -= Math.log(p[i * VV + s[i + 1]]); } }
  }
  const pc = (a, b) => ((100 * a) / Math.max(1, b)).toFixed(1) + "%";
  return `names: unseen pairs first ${pc(c.unseen[0], c.unseen[1])} later ${pc(c.unseen[2], c.unseen[3])} (${c.unseen[3]}) | seen pairs first ${pc(c.seen[0], c.seen[1])} later ${pc(c.seen[2], c.seen[3])} (${c.seen[3]}) | repeats: first copy ${pc(r1, n1)}, second copy ${pc(r2, n2)}, loss ${(l2 / n2).toFixed(2)}`;
}
if (process.env.ch7 || process.argv.includes("ch7=1")) { const m = L.loadModel7(process.env.CH7 || "F8_19000.bin"); console.log("chapter 7:", measure((t) => L.ATN.forward(m, t).p, 2000)); }
for (const f of process.argv.slice(2).filter((x) => !x.includes("="))) { const m = L.loadModel(f); console.log(f.padEnd(14), measure((t) => L.GPT.forward(m, t).p, m.V)); }
