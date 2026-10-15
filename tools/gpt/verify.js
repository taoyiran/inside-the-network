// Checks of what ships: gpt/weights.js against the checkpoint (32-bit rounding only), the stored moments of both runs
// against fresh measurements of their checkpoints (every one, or every k-th with every=k), the retrained chapter 7
// network's rates against a fresh measurement, the gradient sizes of Secret 5 against fresh ones, and the logs against
// the runs' log files.
// node verify.js [every=1]
const L = require("./lib.js"), fs = require("fs"), path = require("path");
const ctx = {}; new Function("window", fs.readFileSync(path.join(__dirname, "..", "..", "gpt", "weights.js"), "utf8"))(ctx);
const W = ctx.ITN_GPT, every = +((process.argv.find((a) => a.startsWith("every=")) || "every=1").slice(6));
const m = L.loadModel(`R_${W.steps}.bin`), b = Buffer.from(W.net, "base64"), sizes = L.sizes(W.shape);
let o = 0, worst = 0, rel = 0;
for (const key of L.GPT.KEYS) { if (m[key].length !== sizes[key]) throw new Error("size of " + key); for (let i = 0; i < m[key].length; i++, o += 4) { const v = b.readFloatLE(o), d = Math.abs(v - m[key][i]); worst = Math.max(worst, d); rel = Math.max(rel, d / Math.max(1e-30, Math.abs(m[key][i]))); } }
console.log(`weights.js: ${o / 4} numbers (${o === b.length ? "all of the file" : "FILE SIZE MISMATCH"}), shape ${JSON.stringify(W.shape)}, largest difference from the checkpoint ${worst.toExponential(1)} (relative ${rel.toExponential(1)}: 32-bit rounding)`);
for (const [label, run, moments] of [["moments", "R", W.moments], ["plain moments", W.plain.run, W.plain.moments]]) {
  let mw = 0, n = 0;
  moments.forEach(([s, , names, second], i) => {
    if (i % every && i !== moments.length - 1) return;
    const x = L.pageMeasures(L.loadModel(`${run}_${s}.bin`)); n++;
    mw = Math.max(mw, Math.abs(x.unseen.later - names), Math.abs(x.second - second));
  });
  console.log(`${label}: ${n} of ${moments.length} checked, largest difference ${mw.toExponential(1)} (stored to 4 decimals)`);
}
const R = W.ch7renamed, x7 = L.pageMeasures(L.loadModel7(`${R.run}_${R.steps}.bin`), L.ATN);
console.log(`chapter 7 retrained: never-seen pairs later ${x7.unseen.later.toFixed(4)} (stored ${R.unseenLater}), second copy ${x7.second.toFixed(4)} (stored ${R.second})`);
// the gradient sizes of Secret 5, measured again (L.blockGrads) on the same checkpoints
{
  const G = W.variants.grads, same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  if (!G) console.log("gradient sizes: NOT STORED");
  else { const p = L.blockGrads(L.GPT, L.loadModel(`R_${G.step}.bin`)), n = L.blockGrads(L.variant("nores"), L.loadModel(`${W.variants.nores.run}_${G.step}.bin`)); console.log(`gradient sizes at step ${G.step}: ${same(p, G.plain) && same(n, G.nores) ? "equal to the stored ones" : `DIFFERENT: ${JSON.stringify({ plain: p, nores: n })} against ${JSON.stringify(G)}`}`); }
}
const logOf = (r) => fs.readFileSync(L.work("runs", r + ".txt"), "utf8").split("\n").filter((l) => l.startsWith("step ")).length;
console.log(`logs: ${W.log.length} lines (run file: ${logOf("R")}), plain ${W.plain.log.length}, variants ${W.variants.nores.log.length} and ${W.variants.nonorm.log.length}`);
console.log("tokenizer script identical to chapter 6's:", L.script("tok") === L.W6.script("tok"));
