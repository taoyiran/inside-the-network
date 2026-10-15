// Write gpt/weights.js: the finished GPT, its run's log and moments of training (Secret 3), the same for the run on the
// stories as they are (Secret 3's comparison), the logs of the two variants (Secret 5), and chapter 7's network
// retrained with the same made-up names, measured as the page measures (Secret 1).
// node package.js   (runs: R, the GPT; G, without made-up names; Rnr and Rnn, the variants; ../attention F8R)
const L = require("./lib.js"), fs = require("fs"), path = require("path");
const RUN = "R", FINAL = +(process.env.FINAL || 9500), PLAIN = "G", NORES = "Rnr", NONORM = "Rnn", CH7 = ["F8R", 19000];
const QUICK = +(process.env.QUICK || 1); // for a trial page: measure only every QUICK-th checkpoint
const out = process.env.OUT || path.join(__dirname, "..", "..", "gpt");
const f32 = (arrs) => { const total = arrs.reduce((a, x) => a + x.length, 0), b = Buffer.alloc(total * 4); let o = 0; for (const a of arrs) for (const v of a) { b.writeFloatLE(v, o); o += 4; } return b.toString("base64"); };
const logOf = (r, upto = Infinity) => !fs.existsSync(L.work("runs", r + ".txt")) ? [] : fs.readFileSync(L.work("runs", r + ".txt"), "utf8").split("\n").map((l) => l.match(/^step (\d+) tokens (\d+) .*?validation loss ([\d.]+) accuracy ([\d.]+)%/)).filter(Boolean).map((x) => [+x[1], +x[2], +x[3], +x[4]]).filter((x) => x[0] <= upto);
// every checkpoint of a run, measured as the page measures the finished GPT: [step, tokens seen, never-seen pairs of
// an invented name finished at later mentions, the second copy of repeated tokens, the validation accuracy]
function momentsOf(r, log) {
  const tokensAt = new Map(log.map(([s, t]) => [s, t])), accAt = new Map(log.map(([s, , , a]) => [s, a]));
  return log.map((x) => x[0]).filter((s, i, all) => fs.existsSync(L.work("models", `${r}_${s}.bin`)) && (i % QUICK === 0 || i === all.length - 1)).map((s) => {
    const m = L.loadModel(`${r}_${s}.bin`), x = L.pageMeasures(m);
    console.log(`${r} ${s}: never-seen pairs, later mentions ${(100 * x.unseen.later).toFixed(1)}% | second copy ${(100 * x.second).toFixed(1)}%`);
    return [s, tokensAt.get(s), +x.unseen.later.toFixed(4), +x.second.toFixed(4), accAt.get(s) / 100];
  });
}
const log = logOf(RUN, FINAL), tokens = log.find((x) => x[0] === FINAL)[1];
const moments = momentsOf(RUN, log), plainLog = logOf(PLAIN, 9500), plainMoments = momentsOf(PLAIN, plainLog);
// nores1: without residual connections and without GPT-2's smaller start for Wo and W2 (variant nores1), to check
// that the start is not what holds the network back
const variants = { nores: { run: NORES, log: logOf(NORES) }, nonorm: { run: NONORM, log: logOf(NONORM) }, nores1: { run: NORES + "1", log: logOf(NORES + "1") } };
// how much gradient reaches each block's weights, with and without residual connections, at the last step both were
// trained to (L.blockGrads; grads.js prints it for all three runs)
const vs = variants.nores.log.length ? variants.nores.log[variants.nores.log.length - 1][0] : 0;
if (vs && fs.existsSync(L.work("models", `${RUN}_${vs}.bin`)) && fs.existsSync(L.work("models", `${NORES}_${vs}.bin`))) {
  variants.grads = { step: vs, plain: L.blockGrads(L.GPT, L.loadModel(`${RUN}_${vs}.bin`)), nores: L.blockGrads(L.variant("nores"), L.loadModel(`${NORES}_${vs}.bin`)) };
  console.log("gradient size by block:", JSON.stringify(variants.grads));
}
// chapter 7's network, retrained with the same made-up names
let ch7renamed = null;
if (fs.existsSync(L.A7.work("models", `${CH7[0]}_${CH7[1]}.bin`))) {
  const m7 = L.loadModel7(`${CH7[0]}_${CH7[1]}.bin`), x7 = L.pageMeasures(m7, L.ATN), e7 = L.A7.evaluate(m7, L.storiesOf("test.bin").stories.slice(0, 100));
  ch7renamed = { run: CH7[0], steps: CH7[1], unseenLater: +x7.unseen.later.toFixed(4), wrongPiece: +x7.wrongPiece.toFixed(4), seenLater: +x7.seen.later.toFixed(4), allLater: +x7.all.later.toFixed(4), second: +x7.second.toFixed(4), test: +e7.acc.toFixed(4) };
}
console.log("chapter 7, retrained with made-up names:", JSON.stringify(ch7renamed));
const m = L.loadModel(`${RUN}_${FINAL}.bin`), xm = L.pageMeasures(m);
console.log(`the GPT, as the page measures it: never-seen pairs at later mentions ${(100 * xm.unseen.later).toFixed(1)}% (first mentions ${(100 * xm.unseen.first).toFixed(1)}%), a wrong piece of the name ${(100 * xm.wrongPiece).toFixed(1)}%, second copy ${(100 * xm.second).toFixed(1)}%`);
fs.writeFileSync(path.join(out, "weights.js"), `// The GPT trained in advance with this chapter's own code, on chapter 6's 20,000 TinyStories training stories, with
// half of the stories whose hero has a common one-token name given a made-up name instead (tools/gpt/rename.js): a
// table of 2,000 rows of 64; 4 blocks, each with two layer norms, 4 heads of 16 numbers with a score for each distance
// back, 0 to ${m.T - 1}, the table Wo and an MLP of 256 neurons; a last layer norm; 2,000 scores. Its weights, rounded
// to 32-bit numbers (little-endian, then base64), in the order ${L.GPT.KEYS.join(", ")}, each kind of weight
// block after block.
// log: [step, training tokens seen, loss, accuracy %] measured during the run on the validation stories 1,001 to 1,300
// (other stories than the 1,000 test stories).
// moments: [step, training tokens seen, invented names: never-seen pairs finished at later mentions, repeated tokens:
// the second copy guessed right, validation accuracy 0 to 1], measured as the page measures the finished GPT, with
// the checkpoints saved during the same run. plain: the same for the same network trained on the stories as they are.
// variants: the logs of the same network trained without residual connections and without layer norms.
// ch7renamed: chapter 7's network retrained with the same made-up names, measured as this page measures the GPT.
window.ITN_GPT = {
  shape: ${JSON.stringify(L.shapeOf(m))},
  steps: ${FINAL},
  tokens: ${tokens},
  net: "${f32(L.GPT.KEYS.map((k) => m[k]))}",
  log: ${JSON.stringify(log)},
  moments: ${JSON.stringify(moments)},
  plain: ${JSON.stringify({ run: PLAIN, log: plainLog, moments: plainMoments })},
  variants: ${JSON.stringify(variants)},
  ch7renamed: ${JSON.stringify(ch7renamed)},
};
`);
console.log("weights.js", fs.statSync(path.join(out, "weights.js")).size, "bytes |", moments.length, "moments,", plainMoments.length, "plain moments");
