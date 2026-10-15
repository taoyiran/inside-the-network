// Train with the page's own trainStep on one thread, drawing stories exactly as ptrain.js does.
// node train.js steps=20 save=T1   then compare with ptrain.js's checkpoint: node train.js cmp=A_20.bin ...
const L = require("./lib.js"), RN = require("../gpt/rename.js");
const DEF = { T: 256, d: 32, H: 8, dk: 8, dv: 16, h: 128, lr: 0.008, B: 8, steps: 20, decay: 0.3, seed: 1, save: "T", cmp: "", sched: 0, rename: 0 };
const a = L.args(DEF), cfg = { V: 2000, T: a.T, d: a.d, H: a.H, dk: a.dk, dv: a.dv, h: a.h };
const { tok, stories, starts } = L.storiesOf("train.bin");
const storyAt = (t) => L.storyAt(starts, t);
const rand = L.mulberry32(a.seed), m = L.ATN.makeModel(cfg, rand), adam = L.ATN.makeAdam(m), g = L.ATN.zeros(m);
const total = a.sched || a.steps; // sched: the length of the schedule, when stopping early to compare with a longer run
const t0 = Date.now();
for (let s = 1; s <= a.steps; s++) {
  const batch = []; for (let b = 0; b < a.B; b++) { const st = stories[storyAt(Math.floor(rand() * tok.length))]; batch.push(a.rename ? RN.applyRename(st, a.rename, RN.draw(rand)) : st); }
  const rate = s > total * (1 - a.decay) ? (a.lr * (total - s + 1)) / (total * a.decay) : a.lr;
  const loss = L.ATN.trainStep(m, batch, rate, adam, g);
  if (s % 5 === 0) console.log(`step ${s} loss ${loss.toFixed(4)} | ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
L.saveModel(m, `${a.save}_${a.steps}.bin`);
if (a.cmp) { const o = L.loadModel(a.cmp); let worst = 0; for (const key of L.ATN.KEYS) for (let i = 0; i < m[key].length; i++) worst = Math.max(worst, Math.abs(m[key][i] - o[key][i])); console.log("largest weight difference against", a.cmp, worst); }
