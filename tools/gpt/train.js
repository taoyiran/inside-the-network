// Train with the page's own trainStep on one thread, drawing stories exactly as ptrain.js does.
// node train.js steps=10 save=T1   then compare with ptrain.js's checkpoint: node train.js steps=10 cmp=P_10.bin ...
const L = require("./lib.js"), RN = require("./rename.js");
const DEF = { T: 256, d: 64, L: 4, H: 4, M: 256, lr: 0.004, B: 16, steps: 10, decay: 0.3, warm: 0, seed: 1, save: "T", cmp: "", sched: 0, rename: 0 };
const a = L.args(DEF), cfg = { V: 2000, T: a.T, d: a.d, L: a.L, H: a.H, M: a.M };
const { tok, stories, starts } = L.storiesOf("train.bin");
const rand = L.mulberry32(a.seed), m = L.GPT.makeModel(cfg, rand), adam = L.GPT.makeAdam(m), g = L.GPT.zeros(m), one = L.GPT.zeros(m);
const total = a.sched || a.steps; // sched: the length of the schedule, when stopping early to compare with a longer run
const t0 = Date.now();
for (let s = 1; s <= a.steps; s++) {
  const batch = []; for (let b = 0; b < a.B; b++) { const st = stories[L.storyAt(starts, Math.floor(rand() * tok.length))]; batch.push(a.rename ? RN.applyRename(st, a.rename, RN.draw(rand)) : st); }
  const rate = s <= a.warm ? (a.lr * s) / a.warm : s > total * (1 - a.decay) ? (a.lr * (total - s + 1)) / (total * a.decay) : a.lr;
  const loss = L.GPT.trainStep(m, batch, rate, adam, g, one);
  console.log(`step ${s} loss ${loss.toFixed(4)} | ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
L.saveModel(m, `${a.save}_${a.steps}.bin`);
if (a.cmp) { const o = L.loadModel(a.cmp); let worst = 0, n = 0; for (const key of L.GPT.KEYS) for (let i = 0; i < m[key].length; i++) { worst = Math.max(worst, Math.abs(m[key][i] - o[key][i])); if (m[key][i] !== o[key][i]) n++; } console.log("against", a.cmp, ": largest weight difference", worst, ",", n, "weights differ in any bit"); }
