// Data-parallel training with the page's own forward and backward, split across worker threads.
// Each step: the main thread picks the batch exactly as the page's trainer does (mulberry32(seed): the starting
// weights, then 64 random positions per step); each worker runs forward and backward on its share of the batch,
// into its own gradient buffer; then each worker adds up all the workers' gradients for its slice of the weights
// and applies the page's Adam update to that slice. Only the order of the additions differs from the page's
// trainStep; with workers=1 the result is bit-identical to it.
// The finished network: node ptrain.js k=8 h=128 lr=0.001 steps=600000 decay=0.3 workers=8 save=Q128
const { Worker, isMainThread, workerData } = require("worker_threads");
const L = require("./lib.js");
if (isMainThread) {
  const { k, d, h, lr, B, steps, decay, seed, every, save, workers, stop } = L.args({ k: 8, d: 32, h: 128, lr: 0.001, B: 64, steps: 600000, decay: 0.3, seed: 1, every: 10000, save: "P", workers: 8, stop: 0 }); // stop: end early, keeping the schedule of `steps`
  const tok = L.loadTokens("train.bin"), st = L.starts(tok), te = L.loadTokens("test.bin"), tst = L.starts(te);
  const rand = L.mulberry32(seed), m0 = L.NWM.makeModel({ V: 2000, k, d, h }, rand);
  const sizes = L.NWM.KEYS.map((key) => m0[key].length), P = sizes.reduce((a, b) => a + b, 0);
  const sab = (n, T = Float64Array) => new SharedArrayBuffer(n * T.BYTES_PER_ELEMENT);
  const shared = { w: sab(P), am: sab(P), as: sab(P), g: sab(P * workers), ctx: sab(B * k, Int32Array), next: sab(B, Int32Array), ctrl: sab(8, Int32Array), loss: sab(workers), lr: sab(1) };
  const w = new Float64Array(shared.w); let o = 0; for (const key of L.NWM.KEYS) { w.set(m0[key], o); o += m0[key].length; }
  const ctxA = new Int32Array(shared.ctx), nextA = new Int32Array(shared.next), ctrl = new Int32Array(shared.ctrl), lrA = new Float64Array(shared.lr);
  for (let i = 0; i < workers; i++) new Worker(__filename, { workerData: { i, workers, shared, cfg: { V: 2000, k, d, h }, B, sizes } });
  // run one phase on every worker and wait for all of them
  const phase = (kind, step = 0, rate = 0) => {
    ctrl[2] = kind; ctrl[3] = step; lrA[0] = rate; Atomics.store(ctrl, 1, 0);
    Atomics.add(ctrl, 0, 1); Atomics.notify(ctrl, 0);
    for (;;) { const done = Atomics.load(ctrl, 1); if (done === workers) break; Atomics.wait(ctrl, 1, done); }
  };
  const view = () => { const m = { V: 2000, k, d, h }; let q = 0; L.NWM.KEYS.forEach((key, i) => { m[key] = w.subarray(q, q + sizes[i]); q += sizes[i]; }); return m; };
  // choosing among networks uses the validation stories 1,001 to 2,000, never the 1,000 test stories the page uses
  let half = 0, n = 0; while (n < 1000) { if (te[half] === 0) n++; half++; }
  const valPos = Array.from({ length: 20000 }, (_, i) => half + Math.floor((i * (te.length - half)) / 20000));
  const t0 = Date.now();
  setTimeout(() => {
    for (let s = 1; s <= (stop || steps); s++) {
      for (let b = 0; b < B; b++) { const t = Math.floor(rand() * tok.length); ctxA.set(L.ctxAt(tok, st, t, k), b * k); nextA[b] = tok[t]; }
      const rate = s > steps * (1 - decay) ? (lr * (steps - s + 1)) / (steps * decay) : lr; // the last part: slow down to 0
      phase(1); phase(2, s, rate);
      if (s % every === 0 || s === 1000 || s === 3000) { const e = L.evaluate(view(), te, tst, valPos); L.saveModel(view(), `${save}_${s}.bin`); console.log(`step ${s} ex ${s * B} validation loss ${e.loss.toFixed(3)} accuracy ${(100 * e.acc).toFixed(1)}% | ${((Date.now() - t0) / 1000).toFixed(0)}s`); }
    }
    phase(9); process.exit(0);
  }, 300);
} else {
  const { i, workers, shared, cfg, B, sizes } = workerData;
  const P = sizes.reduce((a, b) => a + b, 0), k = cfg.k;
  const w = new Float64Array(shared.w), am = new Float64Array(shared.am), as = new Float64Array(shared.as), gAll = new Float64Array(shared.g);
  const ctxA = new Int32Array(shared.ctx), nextA = new Int32Array(shared.next), ctrl = new Int32Array(shared.ctrl), lrA = new Float64Array(shared.lr);
  const split = (buf, base) => { const m = {}; let q = base; L.NWM.KEYS.forEach((key, j) => { m[key] = buf.subarray(q, q + sizes[j]); q += sizes[j]; }); return m; };
  const model = { ...cfg, ...split(w, 0) }, g = split(gAll, i * P), mine = gAll.subarray(i * P, (i + 1) * P);
  const lo = Math.floor((i * P) / workers), hi = Math.floor(((i + 1) * P) / workers);
  let gen = 0;
  for (;;) {
    Atomics.wait(ctrl, 0, gen); gen = Atomics.load(ctrl, 0);
    const kind = ctrl[2];
    if (kind === 9) { Atomics.add(ctrl, 1, 1); Atomics.notify(ctrl, 1); break; }
    if (kind === 1) {
      mine.fill(0);
      for (let b = i; b < B; b += workers) { const context = Array.from(ctxA.subarray(b * k, (b + 1) * k)), next = nextA[b]; L.NWM.backward(model, context, next, L.NWM.forward(model, context), g); }
    } else if (kind === 2) {
      // the page's Adam step, on this worker's slice of the weights
      const t = ctrl[3], lr = lrA[0], c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      for (let j = lo; j < hi; j++) {
        let s = 0; for (let q = 0; q < workers; q++) s += gAll[q * P + j];
        const slope = s / B;
        am[j] = 0.9 * am[j] + 0.1 * slope;
        as[j] = 0.999 * as[j] + 0.001 * slope * slope;
        w[j] -= (lr * am[j]) / c1 / (Math.sqrt(as[j] / c2) + 1e-8);
      }
    }
    Atomics.add(ctrl, 1, 1); Atomics.notify(ctrl, 1);
  }
}
