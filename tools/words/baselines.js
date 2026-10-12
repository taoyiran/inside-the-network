// Counting what followed the last n tokens (falling back to fewer when unseen), on every token of the first
// 100 test stories, split by whether the last 4 tokens appear together in the training stories; and, given
// checkpoint names, the same for those networks. This is what the page measures live.
// node baselines.js Q128_600000.bin
const L = require("./lib.js");
const tok = L.loadTokens("train.bin"), st = L.starts(tok), te = L.loadTokens("test.bin"), tst = L.starts(te), K = 8;
let last = 0, n = 0; while (n < 100) { if (te[last] === 0) n++; last++; }
const pos = Array.from({ length: last }, (_, i) => i), key = (c, l) => c.slice(K - l).join(",");
const want = Array.from({ length: K + 1 }, () => new Map());
for (const t of pos) { const c = L.ctxAt(te, tst, t, K); for (let l = 1; l <= K; l++) if (!want[l].has(key(c, l))) want[l].set(key(c, l), new Map()); }
for (let t = 0; t < tok.length; t++) { const c = L.ctxAt(tok, st, t, K); for (let l = 1; l <= K; l++) { const m = want[l].get(key(c, l)); if (!m) break; m.set(tok[t], (m.get(tok[t]) || 0) + 1); } }
const best = (m) => { let b = -1, bn = 0; for (const [x, v] of m) if (v > bn || (v === bn && x < b)) { bn = v; b = x; } return b; };
const uni = (() => { const c = new Map(); for (const x of tok) c.set(x, (c.get(x) || 0) + 1); return best(c); })();
const nets = process.argv.slice(2).filter((a) => !a.includes("=")).map((f) => [f, L.loadModel(f)]);
const rows = {}, add = (cat, col, ok) => { rows[cat] = rows[cat] || { n: 0 }; if (col === "n") rows[cat].n++; else rows[cat][col] = (rows[cat][col] || 0) + ok; };
for (const t of pos) {
  const c = L.ctxAt(te, tst, t, K); let seen = 0; for (let l = 1; l <= K; l++) { if (want[l].get(key(c, l)).size) seen = l; else break; }
  const guess = (l) => (Math.min(l, seen) ? best(want[Math.min(l, seen)].get(key(c, Math.min(l, seen)))) : uni);
  for (const cat of ["all", seen >= 4 ? "seen4" : "new4", seen >= 8 ? "seen8" : "new8"]) {
    add(cat, "n"); for (const l of [0, 1, 4, 8]) add(cat, "count" + l, Number(guess(l) === te[t]));
    for (const [f, m] of nets) { const p = L.NWM.forward(m, c).p; let b = 0; for (let j = 1; j < p.length; j++) if (p[j] > p[b]) b = j; add(cat, f, Number(b === te[t])); }
  }
}
for (const [cat, r] of Object.entries(rows)) console.log(cat.padEnd(6), String(r.n).padStart(6), "tokens |", Object.entries(r).filter(([k]) => k !== "n").map(([k, v]) => `${k} ${(100 * v / r.n).toFixed(1)}%`).join("  "));
