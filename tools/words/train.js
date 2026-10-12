// Train with the page's own trainStep on one thread: the page's trainer, step for step (seed 1 = the page's live run).
// node train.js steps=1000 lr=0.001 save=T  (decay: the share of the steps at the end over which the rate falls to 0)
const L = require("./lib.js");
const { k, d, h, lr, B, steps, decay, seed, every, save } = L.args({ k: 8, d: 32, h: 128, lr: 0.001, B: 64, steps: 1000, decay: 0.3, seed: 1, every: 1000, save: "T" });
const tok = L.loadTokens("train.bin"), st = L.starts(tok), te = L.loadTokens("test.bin"), tst = L.starts(te);
const rand = L.mulberry32(seed), m = L.NWM.makeModel({ V: 2000, k, d, h }, rand), adam = L.NWM.makeAdam(m), g = L.NWM.zeros(m);
let half = 0, n = 0; while (n < 1000) { if (te[half] === 0) n++; half++; }
const valPos = Array.from({ length: 20000 }, (_, i) => half + Math.floor((i * (te.length - half)) / 20000));
const t0 = Date.now();
for (let s = 1; s <= steps; s++) {
  const batch = [];
  for (let b = 0; b < B; b++) { const t = Math.floor(rand() * tok.length); batch.push({ context: L.ctxAt(tok, st, t, k), next: tok[t] }); }
  const rate = s > steps * (1 - decay) ? (lr * (steps - s + 1)) / (steps * decay) : lr;
  L.NWM.trainStep(m, batch, rate, adam, g);
  if (s % every === 0) { const e = L.evaluate(m, te, tst, valPos); L.saveModel(m, `${save}_${s}.bin`); console.log(`step ${s} validation loss ${e.loss.toFixed(3)} accuracy ${(100 * e.acc).toFixed(1)}% | ${((Date.now() - t0) / 1000).toFixed(0)}s`); }
}
