// Data-parallel training of the attention network with the page's own forward, backward and Adam, on worker threads.
// Each step: the main thread draws B stories exactly as the page's trainer does (mulberry32(seed): the starting
// weights, then per story a random training token, whose story is taken whole, so long stories come up more
// often, in proportion to their length). All the positions of those stories are cut into one run and shared out
// evenly among the workers; a worker runs forward (with `from`) and backward on its slices of stories, into its
// own gradient buffer. Then each worker adds up all the workers' gradients for its slice of the weights and applies
// the page's Adam update to it. Only the order of the additions differs from the page's trainStep.
// The finished network: node ptrain.js lr=0.008 steps=19000 save=F8 > work/runs/F8.txt
const { Worker, isMainThread, workerData } = require("worker_threads");
const L = require("./lib.js");
const EARLY = [50, 100, 200, 500]; // also saved, for the early moments of training
const DEF = { T: 256, d: 32, H: 8, dk: 8, dv: 16, h: 128, lr: 0.008, B: 8, steps: 19000, decay: 0.3, seed: 1, every: 1000, save: "A", workers: 16, stop: 0, val: 300 };
if (isMainThread) {
  const a = L.args(DEF), cfg = { V: 2000, T: a.T, d: a.d, H: a.H, dk: a.dk, dv: a.dv, h: a.h }, W = a.workers;
  const { tok, stories, starts } = L.storiesOf("train.bin");
  const storyAt = (t) => L.storyAt(starts, t);
  const rand = L.mulberry32(a.seed), m0 = L.ATN.makeModel(cfg, rand);
  const sz = L.ATN.KEYS.map((key) => m0[key].length), P = sz.reduce((x, y) => x + y, 0);
  const sab = (n, T = Float64Array) => new SharedArrayBuffer(n * T.BYTES_PER_ELEMENT);
  const MAXSEG = 4 * (a.B + 1);
  const shared = { w: sab(P), am: sab(P), as: sab(P), g: sab(P * W), seg: sab(W * MAXSEG * 3, Int32Array), nseg: sab(W, Int32Array), ctrl: sab(8, Int32Array), stat: sab(W * 3), lr: sab(1), count: sab(1) };
  const w = new Float64Array(shared.w); let o = 0; for (const key of L.ATN.KEYS) { w.set(m0[key], o); o += m0[key].length; }
  const seg = new Int32Array(shared.seg), nseg = new Int32Array(shared.nseg), ctrl = new Int32Array(shared.ctrl), lrA = new Float64Array(shared.lr), countA = new Float64Array(shared.count), stat = new Float64Array(shared.stat);
  // a worker that fails would leave the main thread waiting forever: stop the run instead
  for (let i = 0; i < W; i++) new Worker(__filename, { workerData: { i, W, shared, cfg, sz, MAXSEG, val: a.val } }).on("error", (e) => { console.error(`worker ${i} failed:`, e); process.exit(1); }).on("exit", (code) => { if (code) { console.error(`worker ${i} exited with code ${code}`); process.exit(1); } });
  const phase = (kind, step = 0, rate = 0) => {
    ctrl[2] = kind; ctrl[3] = step; lrA[0] = rate; Atomics.store(ctrl, 1, 0);
    Atomics.add(ctrl, 0, 1); Atomics.notify(ctrl, 0);
    for (;;) { const done = Atomics.load(ctrl, 1); if (done === W) break; Atomics.wait(ctrl, 1, done); }
  };
  const view = () => { const m = { ...cfg }; let q = 0; L.ATN.KEYS.forEach((key, i) => { m[key] = w.subarray(q, q + sz[i]); q += sz[i]; }); return m; };
  const validate = () => { phase(3); let l = 0, r = 0, n = 0; for (let i = 0; i < W; i++) { l += stat[i * 3]; r += stat[i * 3 + 1]; n += stat[i * 3 + 2]; } return { loss: l / n, acc: r / n, n }; };
  console.log(JSON.stringify({ ...a, params: P }));
  const t0 = Date.now(); let tokens = 0, lossSum = 0, lossN = 0;
  setTimeout(() => {
    for (let s = 1; s <= (a.stop || a.steps); s++) {
      const picked = []; for (let b = 0; b < a.B; b++) picked.push(storyAt(Math.floor(rand() * tok.length)));
      // share the positions of the picked stories out evenly: worker i takes positions [i·M/W, (i+1)·M/W) of the run
      const M = picked.reduce((x, st) => x + stories[st].next.length, 0);
      nseg.fill(0); let base = 0;
      for (const st of picked) {
        const n = stories[st].next.length;
        for (let i = 0; i < W; i++) {
          const lo = Math.max(base, Math.floor((i * M) / W)), hi = Math.min(base + n, Math.floor(((i + 1) * M) / W));
          if (hi > lo) { const k = (i * MAXSEG + nseg[i]++) * 3; seg[k] = st; seg[k + 1] = lo - base; seg[k + 2] = hi - base; }
        }
        base += n;
      }
      const rate = s > a.steps * (1 - a.decay) ? (a.lr * (a.steps - s + 1)) / (a.steps * a.decay) : a.lr;
      countA[0] = M; phase(1); phase(2, s, rate);
      tokens += M; for (let i = 0; i < W; i++) { lossSum += stat[i * 3]; } lossN += M;
      if (s % a.every === 0 || s === (a.stop || a.steps) || EARLY.includes(s)) {
        const e = validate(); L.saveModel(view(), `${a.save}_${s}.bin`);
        console.log(`step ${s} tokens ${tokens} train loss ${(lossSum / lossN).toFixed(3)} validation loss ${e.loss.toFixed(3)} accuracy ${(100 * e.acc).toFixed(1)}% | ${((Date.now() - t0) / 1000).toFixed(0)}s`);
        lossSum = 0; lossN = 0;
      }
    }
    phase(9); process.exit(0);
  }, 300);
} else {
  const { i, W, shared, cfg, sz, MAXSEG, val } = workerData;
  const { stories } = L.storiesOf("train.bin");
  // validation: stories 1,001 to 1,000 + val of the validation file, never the page's 1,000 test stories
  const vs = L.storiesOf("test.bin").stories.slice(1000, 1000 + val).filter((_, j) => j % W === i);
  const P = sz.reduce((x, y) => x + y, 0);
  const w = new Float64Array(shared.w), am = new Float64Array(shared.am), as = new Float64Array(shared.as), gAll = new Float64Array(shared.g);
  const seg = new Int32Array(shared.seg), nseg = new Int32Array(shared.nseg), ctrl = new Int32Array(shared.ctrl), lrA = new Float64Array(shared.lr), countA = new Float64Array(shared.count), stat = new Float64Array(shared.stat);
  const split = (buf, base) => { const m = {}; let q = base; L.ATN.KEYS.forEach((key, j) => { m[key] = buf.subarray(q, q + sz[j]); q += sz[j]; }); return m; };
  const model = { ...cfg, ...split(w, 0) }, g = split(gAll, i * P), mine = gAll.subarray(i * P, (i + 1) * P);
  const lo = Math.floor((i * P) / W), hi = Math.floor(((i + 1) * P) / W);
  let gen = 0;
  for (;;) {
    Atomics.wait(ctrl, 0, gen); gen = Atomics.load(ctrl, 0);
    const kind = ctrl[2];
    if (kind === 9) { Atomics.add(ctrl, 1, 1); Atomics.notify(ctrl, 1); break; }
    if (kind === 1) {
      mine.fill(0); let loss = 0;
      for (let s = 0; s < nseg[i]; s++) {
        const k = (i * MAXSEG + s) * 3, { tokens, next } = stories[seg[k]], from = seg[k + 1], to = seg[k + 2];
        const toks = tokens.subarray(0, to), nx = next.subarray(from, to), f = L.ATN.forward(model, toks, from);
        for (let r = 0; r < nx.length; r++) loss -= Math.log(f.p[r * cfg.V + nx[r]]);
        L.ATN.backward(model, toks, nx, f, g);
      }
      stat[i * 3] = loss;
    } else if (kind === 2) {
      const t = ctrl[3], lr = lrA[0], count = countA[0], c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      for (let j = lo; j < hi; j++) {
        let s = 0; for (let q = 0; q < W; q++) s += gAll[q * P + j];
        const slope = s / count;
        am[j] = 0.9 * am[j] + 0.1 * slope;
        as[j] = 0.999 * as[j] + 0.001 * slope * slope;
        w[j] -= (lr * am[j]) / c1 / (Math.sqrt(as[j] / c2) + 1e-8);
      }
    } else if (kind === 3) {
      const e = vs.length ? L.evaluate(model, vs) : { loss: 0, acc: 0, n: 0 };
      stat[i * 3] = e.loss * e.n; stat[i * 3 + 1] = e.acc * e.n; stat[i * 3 + 2] = e.n;
    }
    Atomics.add(ctrl, 1, 1); Atomics.notify(ctrl, 1);
  }
}
