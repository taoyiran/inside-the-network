// Write attention/weights.js: the finished network, the run's log, and the saved moments for Secret 1.
// node package.js F8 19000   (the run's checkpoints in work/models/, its log in work/runs/<run>.txt)
// The stories are chapter 6's (words/stories.js); this chapter ships no data of its own.
const L = require("./lib.js"), fs = require("fs"), path = require("path");
const run = process.argv[2] || "F8", finalStep = +process.argv[3] || 19000, out = process.env.OUT || path.join(__dirname, "..", "..", "attention");
const f32 = (arrs) => { const total = arrs.reduce((a, x) => a + x.length, 0), b = Buffer.alloc(total * 4); let o = 0; for (const a of arrs) for (const v of a) { b.writeFloatLE(v, o); o += 4; } return b.toString("base64"); };
const log = fs.readFileSync(L.work("runs", run + ".txt"), "utf8").split("\n").map((l) => l.match(/^step (\d+) tokens (\d+) .*?validation loss ([\d.]+) accuracy ([\d.]+)%/)).filter(Boolean).map((x) => [+x[1], +x[2], +x[3], +x[4]]).filter((x) => x[0] <= finalStep);
const tokensAt = new Map(log.map(([s, t]) => [s, t]));
if (!tokensAt.has(finalStep)) throw new Error("no log line for step " + finalStep);
const { stories } = L.storiesOf("test.bin"), test = stories.slice(0, 100);
const MOMENT_STEPS = [0, 50, 100, 200, 500, 1000, 3000, 10000].filter((s) => s < finalStep);
const moments = MOMENT_STEPS.map((s) => {
  const m = s ? L.loadModel(`${run}_${s}.bin`) : L.ATN.makeModel(L.loadModel(`${run}_${finalStep}.bin`), L.mulberry32(L.seedOf(run))); // step 0: the run's starting weights
  const { bins, tops } = L.headStats(m, test);
  console.log("moment", s, "head 1 bins", bins.slice(0, 8).join(" "));
  return [s, s ? tokensAt.get(s) : 0, bins, tops];
});
const m = L.loadModel(`${run}_${finalStep}.bin`);
fs.writeFileSync(path.join(out, "weights.js"), `// The attention network trained in advance with this chapter's own code, on chapter 6's 20,000 TinyStories training
// stories: a table of 2,000 rows of 32; 8 heads, each with queries and keys of 8 numbers, values of 16 and a score
// for each distance back, 0 to ${m.T - 1}; 128 tanh neurons reading 160 numbers; 2,000 scores. Its weights, rounded to
// 32-bit numbers (little-endian, then base64): E, Wq, Wk, Wv, R, W1, b1, W2, b2, in that order.
// log: [step, training tokens seen, loss, accuracy %] measured during the run on the validation stories 1,001 to 1,300
// (other stories than the 1,000 test stories).
// moments: [step, training tokens seen, each head's share of weight in each distance bin (0, 1, 2, 3, 4–7, 8–15,
// 16–63, 64–255; head by head), each head's 4 most-read tokens per appearance], measured on the page's 100 test
// stories with the network as it was at that step of the same run; step 0 is the starting weights.
window.ITN_ATTN = {
  shape: ${JSON.stringify({ V: m.V, T: m.T, d: m.d, H: m.H, dk: m.dk, dv: m.dv, h: m.h })},
  steps: ${finalStep},
  tokens: ${tokensAt.get(finalStep)},
  net: "${f32(L.ATN.KEYS.map((k) => m[k]))}",
  log: ${JSON.stringify(log)},
  moments: ${JSON.stringify(moments)},
};
`);
console.log("weights.js", fs.statSync(path.join(out, "weights.js")).size, "bytes | moments", MOMENT_STEPS.join(", "));
