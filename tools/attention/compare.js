// Counting (last 4 tokens, falling back to fewer), chapter 6's network (last 8 tokens) and attention networks, on
// every token of the first 100 test stories (the page's test tokens), split by where the right answer last appeared:
//   near: among the last 8 tokens (chapter 6's network can see it)
//   far:  earlier in the story, but not among the last 8 tokens (only attention can see it)
//   new:  nowhere earlier in the story
// "words" rows keep only answers that are words or word pieces (not marks, digits or the end of a story);
// "far, not top 100" keeps far answers that are not among the 100 most common tokens of the training stories.
// node compare.js A_20000.bin [more checkpoints]   (chapter 6's network: ../words/work/models/Q128_600000.bin)
const L = require("./lib.js");
const K = 8, NSTORIES = +(process.env.NSTORIES || 100);
const tok = L.loadTokens("train.bin"), st = L.starts(tok), te = L.loadTokens("test.bin"), tst = L.starts(te);
let last = 0, n = 0; while (n < NSTORIES) { if (te[last] === 0) n++; last++; }
const key = (c, l) => c.slice(K - l).join(",");
const want = Array.from({ length: 5 }, () => new Map());
for (let t = 0; t < last; t++) { const c = L.ctxAt(te, tst, t, K); for (let l = 1; l <= 4; l++) if (!want[l].has(key(c, l))) want[l].set(key(c, l), new Map()); }
for (let t = 0; t < tok.length; t++) { const c = L.ctxAt(tok, st, t, K); for (let l = 1; l <= 4; l++) { const m = want[l].get(key(c, l)); if (!m) break; m.set(tok[t], (m.get(tok[t]) || 0) + 1); } }
const best = (m) => { let b = -1, bn = 0; for (const [x, v] of m) if (v > bn || (v === bn && x < b)) { bn = v; b = x; } return b; };
const rank = (() => { const c = new Float64Array(2000); for (const x of tok) c[x]++; const r = new Int32Array(2000); Array.from(c.keys()).sort((a, b) => c[b] - c[a]).forEach((x, i) => (r[x] = i)); return r; })();
const uni = (() => { const c = new Map(); for (const x of tok) c.set(x, (c.get(x) || 0) + 1); return best(c); })();
const nwm = L.W6.loadModel(process.env.CH6 || "Q128_600000.bin");
const nets = process.argv.slice(2).filter((x) => !x.includes("=")).map((f) => [f, L.loadModel(f)]);
const argmax = L.argmax;
// attention guesses for every test position, story by story (the story's tokens start with the <end> before it)
const pad = new Uint16Array(last + 1); pad.set(te.subarray(0, last), 1);
const guesses = nets.map(([, m]) => { const g = new Int32Array(last); let a = 0; for (let t = 0; t < last; t++) if (te[t] === 0) { const f = L.ATN.forward(m, pad.subarray(a, t + 1)); for (let r = 0; r <= t - a; r++) g[a + r] = argmax(f.p, r * m.V, m.V); a = t + 1; } return g; });
const rows = {}, add = (cat, col, ok) => { const r = (rows[cat] = rows[cat] || { n: 0 }); if (col === "n") r.n++; else r[col] = (r[col] || 0) + ok; };
for (let t = 0; t < last; t++) {
  const c = L.ctxAt(te, tst, t, K), ans = te[t];
  let seen = 0; for (let l = 1; l <= 4; l++) { if (want[l].get(key(c, l)).size) seen = l; else break; }
  const count4 = seen ? best(want[seen].get(key(c, seen))) : uni;
  let where = "new"; for (let j = t - 1; j >= tst[t]; j--) if (te[j] === ans) { where = t - j <= K ? "near" : "far"; break; }
  const p = L.NWM.forward(nwm, c).p, g6 = argmax(p, 0, p.length);
  const cats = ["all", where]; if (ans >= 25) cats.push("words", "words " + where);
  if (where === "far" && rank[ans] >= 100) cats.push("far, not top 100");
  for (const cat of cats) {
    add(cat, "n"); add(cat, "count4", Number(count4 === ans)); add(cat, "ch6", Number(g6 === ans));
    nets.forEach(([f], i) => add(cat, f, Number(guesses[i][t] === ans)));
  }
}
const N = rows.all.n;
for (const [cat, r] of Object.entries(rows)) console.log(cat.padEnd(10), String(r.n).padStart(6), `(${((100 * r.n) / N).toFixed(1)}%) |`, Object.entries(r).filter(([k]) => k !== "n").map(([k, v]) => `${k} ${((100 * v) / r.n).toFixed(1)}%`).join("  "));
