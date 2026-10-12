// What a network's table learned: the nearest letter-led tokens to some words (the page shows only tokens that are
// mostly whole words, so its lists can differ slightly), and analogy puzzles b − a + c.
// node analyze.js Q128_600000.bin [cat,big,...]
const L = require("./lib.js");
const m = L.loadModel(process.argv[2]), tk = L.tokenizer(), { V, d, E } = m, U = new Float64Array(V * d);
for (let t = 0; t < V; t++) { let n = 0; for (let i = 0; i < d; i++) n += E[t * d + i] ** 2; n = Math.sqrt(n) || 1; for (let i = 0; i < d; i++) U[t * d + i] = E[t * d + i] / n; }
const word = (t) => /^[a-z]/.test(tk.vocab[t]);
function nearest(vec, k, skip = new Set()) {
  let n = 0; for (const v of vec) n += v * v; n = Math.sqrt(n);
  const sc = []; for (let t = 0; t < V; t++) { if (skip.has(t) || !word(t)) continue; let s = 0; for (let i = 0; i < d; i++) s += U[t * d + i] * vec[i]; sc.push([s / n, t]); }
  return sc.sort((a, b) => b[0] - a[0]).slice(0, k);
}
const row = (t) => Array.from(U.subarray(t * d, (t + 1) * d)), show = (l) => l.map(([s, t]) => `${tk.vocab[t]} ${s.toFixed(2)}`).join(", ");
for (const w of (process.argv[3] || "cat,dog,big,small,up,happy,sad,mom,ran,red,park,lily,said,saw,can,fly,left,rose").split(",")) {
  const t = tk.id.get(w); console.log(w.padEnd(7), t === undefined ? "(not one token)" : show(nearest(row(t), 8, new Set([t]))));
}
const puzzles = [["he", "she", "boy", "girl"], ["dad", "mom", "brother", "sister"], ["his", "her", "he", "she"], ["boy", "girl", "dad", "mom"], ["run", "ran", "walk", "walked"], ["play", "played", "jump", "jumped"], ["go", "went", "see", "saw"], ["is", "was", "are", "were"], ["find", "found", "see", "saw"], ["dog", "dogs", "bird", "birds"], ["toy", "toys", "tree", "trees"]];
let ex = 0, all = 0;
for (const [a, b, c, want] of puzzles) {
  const [ia, ib, ic, iw] = [a, b, c, want].map((w) => tk.id.get(w)), v = row(ib).map((x, i) => x - row(ia)[i] + row(ic)[i]);
  const e = nearest(v, 3, new Set([ia, ib, ic])), f = nearest(v, 3); if (e[0][1] === iw) ex++; if (f[0][1] === iw) all++;
  console.log(`${b} − ${a} + ${c} (${want}): leaving out the three: ${show(e)} | all: ${show(f)}`);
}
console.log(`right first: ${ex} of ${puzzles.length} leaving out the three words, ${all} of ${puzzles.length} without`);
