  // ---------- training (the same settings as the trained networks) ----------
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gaussian(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  // the network reads each spectrogram value minus 0.5: most of a spectrogram is mid-level background,
  // and centering it helps training a little, mostly early on
  const toInput = (spec) => { const g = grid(1, BANDS, FRAMES); for (let i = 0; i < g.d.length; i++) g.d[i] = spec.d[i] - 0.5; return g; };
  // weights scaled by √(2 / inputs) (√(1 / inputs) for the last layer); layer 1's start 50 times smaller, with biases of 0.1,
  // as in chapter 5, so its stencils begin as almost-blank noise and every one of them starts switched on
  function makeCNN(seed) {
    const r = mulberry32(seed);
    const conv = (n, c) => ({ type: "conv", n, k: 5, W: Float64Array.from({ length: n * c * 25 }, () => gaussian(r) * Math.sqrt(2 / (c * 25))), b: new Float64Array(n) });
    const net = [conv(8, 1), { type: "relu" }, { type: "pool" }, conv(16, 8), { type: "relu" }, { type: "pool" }, { type: "dense", n: 10, W: Float64Array.from({ length: 4000 }, () => gaussian(r) * Math.sqrt(1 / 400)), b: new Float64Array(10) }];
    net[0].W.forEach((v, i) => { net[0].W[i] = v * 0.02; }); net[0].b.fill(0.1);
    return net;
  }
  function makePlain(seed) {
    const r = mulberry32(seed);
    return [{ type: "dense", n: 8, W: Float64Array.from({ length: 8192 }, () => gaussian(r) * Math.sqrt(2 / 1024)), b: new Float64Array(8) }, { type: "relu" }, { type: "dense", n: 10, W: Float64Array.from({ length: 80 }, () => gaussian(r) * Math.sqrt(1 / 8)), b: new Float64Array(10) }];
  }
  // move a spectrogram right by dt columns (later) and up by dp bands (higher); spots that come in from outside
  // repeat the nearest edge, as if the background went on
  function moved(spec, dt, dp) {
    const o = grid(1, BANDS, FRAMES);
    for (let r = 0; r < BANDS; r++) for (let c = 0; c < FRAMES; c++) o.d[r * FRAMES + c] = spec.d[Math.max(0, Math.min(BANDS - 1, r + dp)) * FRAMES + Math.max(0, Math.min(FRAMES - 1, c - dt))];
    return o;
  }
  function shuffled(n, r) { const a = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
  // a trainer: passes over the training clips in a fresh random order each time, batches of 20,
  // learning rate 0.1 for the first 8 passes and 0.03 after. With shift above 0, each training picture is first
  // moved up or down by a random number of bands, up to shift, so the network sees every word at many pitches.
  function trainer(seed, make = makeCNN, shift = 0) {
    const T = { net: make(seed), r: mulberry32(6 + seed), rs: mulberry32(99 + seed), step: 0, order: null, pos: 0, pass: 0 };
    T.next = () => {
      if (!T.order || T.pos >= T.order.length) { T.order = shuffled(DATA.train.n, T.r); T.pos = 0; T.pass++; }
      const batch = T.order.slice(T.pos, T.pos + 20).map((k) => {
        const s = specOf(DATA.train, k);
        return { image: toInput(shift ? moved(s, 0, Math.floor(T.rs() * (2 * shift + 1)) - shift) : s), label: DATA.train.labels[k] };
      });
      T.pos += 20; T.step++;
      return trainStep(T.net, batch, T.pass <= 8 ? 0.1 : 0.03);
    };
    return T;
  }
