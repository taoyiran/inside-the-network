// Data-parallel training of the GPT with the page's own forward, backward and Adam, on worker threads.
// Each step: the main thread draws B stories exactly as the page's trainer does (mulberry32(seed): the starting
// weights, then per story a random training token, whose story is taken whole, so long stories come up more often,
// in proportion to their length). The workers take the stories one at a time, longest first, as each becomes free,
// and work out each story's gradient into a slot of its own. Then each worker adds up the B slots, in the batch's
// order, for its share of the weights, and applies the page's Adam update to it. These are the same additions in
// the same order as the page's trainStep, so the result is bit-identical to it for any number of workers.
// The finished network (run R): node ptrain.js lr=0.004 B=16 steps=9500 every=250 early=0,25,50,100,150,200 rename=0.5 save=R
// > work/runs/R.txt; the same without made-up names (run G, Secret 3): the same without rename=0.5, save=G. The full
// list of runs is in docs/chapter-8-training.md.
// variant=nores or variant=nonorm trains the same network without residual connections or without layer norms.
// rename=p gives the hero of some training stories a made-up name (rename.js); then each story also draws a roll and
// a seed, after the token that picks it.
const { Worker, isMainThread, workerData } = require("worker_threads");
const L = require("./lib.js"), RN = require("./rename.js");
const DEF = { T: 256, d: 64, L: 4, H: 4, M: 256, lr: 0.004, B: 16, steps: 10000, decay: 0.3, warm: 0, seed: 1, every: 500, early: "", save: "G", workers: 16, stop: 0, val: 300, variant: "plain", rename: 0 };
if (isMainThread) {
  const a = L.args(DEF), cfg = { V: 2000, T: a.T, d: a.d, L: a.L, H: a.H, M: a.M }, W = a.workers, C = L.variant(a.variant);
  const { tok, stories, starts } = L.storiesOf("train.bin");
  const rand = L.mulberry32(a.seed), m0 = C.makeModel(cfg, rand);
  const sz = C.KEYS.map((key) => m0[key].length), P = sz.reduce((x, y) => x + y, 0);
  const sab = (n, T = Float64Array) => new SharedArrayBuffer(n * T.BYTES_PER_ELEMENT);
  const shared = { w: sab(P), am: sab(P), as: sab(P), g: sab(P * a.B), batch: sab(a.B, Int32Array), order: sab(a.B, Int32Array), ctrl: sab(8, Int32Array), loss: sab(a.B), stat: sab(W * 3), lr: sab(1), count: sab(1), roll: sab(a.B), rseed: sab(a.B, Uint32Array), len: sab(a.B, Int32Array) };
  const w = new Float64Array(shared.w); let o = 0; for (const key of C.KEYS) { w.set(m0[key], o); o += m0[key].length; }
  const batch = new Int32Array(shared.batch), order = new Int32Array(shared.order), ctrl = new Int32Array(shared.ctrl), lrA = new Float64Array(shared.lr), countA = new Float64Array(shared.count), lossA = new Float64Array(shared.loss), stat = new Float64Array(shared.stat);
  const rollA = new Float64Array(shared.roll), rseedA = new Uint32Array(shared.rseed), lenA = new Int32Array(shared.len);
  // a worker that fails would leave the main thread waiting forever: stop the run instead
  for (let i = 0; i < W; i++) new Worker(__filename, { workerData: { i, W, shared, cfg, sz, B: a.B, val: a.val, variant: a.variant, rename: a.rename } }).on("error", (e) => { console.error(`worker ${i} failed:`, e); process.exit(1); }).on("exit", (code) => { if (code) { console.error(`worker ${i} exited with code ${code}`); process.exit(1); } });
  const phase = (kind, step = 0, rate = 0) => {
    ctrl[2] = kind; ctrl[3] = step; lrA[0] = rate; Atomics.store(ctrl, 1, 0); Atomics.store(ctrl, 4, 0);
    Atomics.add(ctrl, 0, 1); Atomics.notify(ctrl, 0);
    for (;;) { const done = Atomics.load(ctrl, 1); if (done === W) break; Atomics.wait(ctrl, 1, done); }
  };
  const view = () => { const m = { ...cfg }; let q = 0; C.KEYS.forEach((key, i) => { m[key] = w.subarray(q, q + sz[i]); q += sz[i]; }); return m; };
  const validate = () => { phase(3); let l = 0, r = 0, n = 0; for (let i = 0; i < W; i++) { l += stat[i * 3]; r += stat[i * 3 + 1]; n += stat[i * 3 + 2]; } return { loss: l / n, acc: r / n, n }; };
  const early = String(a.early).split(",").filter(Boolean).map(Number);
  console.log(JSON.stringify({ ...a, params: P }));
  const t0 = Date.now(); let tokens = 0, lossSum = 0, lossN = 0;
  const save = (s) => { const e = validate(); L.saveModel(view(), `${a.save}_${s}.bin`); console.log(`step ${s} tokens ${tokens} train loss ${(lossSum / Math.max(1, lossN)).toFixed(3)} validation loss ${e.loss.toFixed(3)} accuracy ${(100 * e.acc).toFixed(1)}% | ${((Date.now() - t0) / 1000).toFixed(0)}s`); lossSum = 0; lossN = 0; };
  setTimeout(() => {
    if (early.includes(0)) save(0);
    for (let s = 1; s <= (a.stop || a.steps); s++) {
      for (let b = 0; b < a.B; b++) { batch[b] = L.storyAt(starts, Math.floor(rand() * tok.length)); if (a.rename) [rollA[b], rseedA[b]] = RN.draw(rand); }
      Array.from(batch.keys()).sort((x, y) => stories[batch[y]].next.length - stories[batch[x]].next.length || x - y).forEach((b, k) => (order[k] = b));
      // the learning rate: an optional warm-up from 0, then constant, then falling linearly to 0 over the last `decay`
      const rate = s <= a.warm ? (a.lr * s) / a.warm : s > a.steps * (1 - a.decay) ? (a.lr * (a.steps - s + 1)) / (a.steps * a.decay) : a.lr;
      phase(1); let M = 0; for (let b = 0; b < a.B; b++) M += lenA[b]; // the positions of the step's stories, as read
      countA[0] = M; phase(2, s, rate);
      tokens += M; for (let b = 0; b < a.B; b++) lossSum += lossA[b]; lossN += M;
      if (s % a.every === 0 || s === (a.stop || a.steps) || early.includes(s)) save(s);
    }
    phase(9); process.exit(0);
  }, 300);
} else {
  const { i, W, shared, cfg, sz, B, val, variant, rename } = workerData, C = L.variant(variant);
  const { stories } = L.storiesOf("train.bin");
  // validation: stories 1,001 to 1,000 + val of the validation file, never the page's 1,000 test stories
  const vs = L.storiesOf("test.bin").stories.slice(1000, 1000 + val).filter((_, j) => j % W === i);
  const P = sz.reduce((x, y) => x + y, 0);
  const w = new Float64Array(shared.w), am = new Float64Array(shared.am), as = new Float64Array(shared.as), gAll = new Float64Array(shared.g);
  const batch = new Int32Array(shared.batch), order = new Int32Array(shared.order), ctrl = new Int32Array(shared.ctrl), lrA = new Float64Array(shared.lr), countA = new Float64Array(shared.count), lossA = new Float64Array(shared.loss), stat = new Float64Array(shared.stat);
  const rollA = new Float64Array(shared.roll), rseedA = new Uint32Array(shared.rseed), lenA = new Int32Array(shared.len);
  const split = (buf, base) => { const m = { L: cfg.L }; let q = base; C.KEYS.forEach((key, j) => { m[key] = buf.subarray(q, q + sz[j]); q += sz[j]; }); return m; };
  const model = { ...cfg, ...split(w, 0) };
  const lo = Math.floor((i * P) / W), hi = Math.floor(((i + 1) * P) / W);
  let gen = 0;
  for (;;) {
    Atomics.wait(ctrl, 0, gen); gen = Atomics.load(ctrl, 0);
    const kind = ctrl[2];
    if (kind === 9) { Atomics.add(ctrl, 1, 1); Atomics.notify(ctrl, 1); break; }
    if (kind === 1) {
      // take the next story not yet taken, longest first
      for (let k = Atomics.add(ctrl, 4, 1); k < B; k = Atomics.add(ctrl, 4, 1)) {
        const b = order[k], slot = gAll.subarray(b * P, (b + 1) * P), g = split(gAll, b * P), { tokens, next } = RN.applyRename(stories[batch[b]], rename, [rollA[b], rseedA[b]]);
        slot.fill(0);
        const f = C.forward(model, tokens); let loss = 0;
        for (let r = 0; r < next.length; r++) loss -= Math.log(f.p[r * cfg.V + next[r]]);
        C.backward(model, tokens, next, f, g);
        lossA[b] = loss; lenA[b] = next.length;
      }
    } else if (kind === 2) {
      const t = ctrl[3], lr = lrA[0], count = countA[0], c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      for (let j = lo; j < hi; j++) {
        let s = 0; for (let b = 0; b < B; b++) s += gAll[b * P + j];
        const slope = s / count;
        am[j] = 0.9 * am[j] + 0.1 * slope;
        as[j] = 0.999 * as[j] + 0.001 * slope * slope;
        w[j] -= (lr * am[j]) / c1 / (Math.sqrt(as[j] / c2) + 1e-8);
      }
    } else if (kind === 3) {
      const e = vs.length ? L.evaluate(model, vs, C) : { loss: 0, acc: 0, n: 0 };
      stat[i * 3] = e.loss * e.n; stat[i * 3 + 1] = e.acc * e.n; stat[i * 3 + 2] = e.n;
    }
    Atomics.add(ctrl, 1, 1); Atomics.notify(ctrl, 1);
  }
}
