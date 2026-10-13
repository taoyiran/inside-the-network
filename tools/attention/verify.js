// Checks of what ships: attention/weights.js against the checkpoint (32-bit rounding only), every saved moment against
// a fresh measurement of its checkpoint (as package.js measures it), and the page's tokenizer against chapter 6's (the
// page carries a copy).
// node verify.js F8 19000
const L = require("./lib.js"), fs = require("fs"), path = require("path");
const run = process.argv[2] || "F8", step = +process.argv[3] || 19000;
const ctx = {}; new Function("window", fs.readFileSync(path.join(__dirname, "..", "..", "attention", "weights.js"), "utf8"))(ctx);
const W = ctx.ITN_ATTN, m = L.loadModel(`${run}_${step}.bin`), b = Buffer.from(W.net, "base64");
let o = 0, worst = 0, rel = 0;
for (const key of L.ATN.KEYS) for (let i = 0; i < m[key].length; i++, o += 4) { const v = b.readFloatLE(o), d = Math.abs(v - m[key][i]); worst = Math.max(worst, d); rel = Math.max(rel, d / Math.max(1e-30, Math.abs(m[key][i]))); }
console.log(`weights.js: ${o / 4} numbers (${o === b.length ? "all of the file" : "FILE SIZE MISMATCH"}), shape ${JSON.stringify(W.shape)}, largest difference from the checkpoint ${worst.toExponential(1)} (relative ${rel.toExponential(1)}: 32-bit rounding)`);
console.log(`steps ${W.steps}, tokens ${W.tokens}, ${W.log.length} log lines, moments at ${W.moments.map((x) => x[0]).join(", ")}`);
const test = L.storiesOf("test.bin").stories.slice(0, 100);
let mw = 0, tops = true;
for (const [s, , bins, tp] of W.moments) {
  const mm = s ? L.loadModel(`${run}_${s}.bin`) : L.ATN.makeModel(m, L.mulberry32(L.seedOf(run))), st = L.headStats(mm, test);
  bins.forEach((v, i) => (mw = Math.max(mw, Math.abs(v - st.bins[i])))); if (st.tops.join() !== tp.join()) tops = false;
}
console.log(`moments: largest difference in the distance shares ${mw.toExponential(1)} (stored to 4 decimals), most-read tokens ${tops ? "identical" : "DIFFERENT"}`);
const tok7 = L.script("tok"), tok6 = L.W6.script("tok");
console.log("tokenizer script identical to chapter 6's:", tok7 === tok6);
