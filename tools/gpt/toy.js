// The live experiment of Secret 3, in Node, for several seeds: a GPT of 2 blocks, 2 heads each, rows of 32 and no MLP
// neurons (M = 0: each block keeps only the MLP's bias, a learned row added at every position), trained (16 sequences
// per step, Adam at 0.01) on <end>, then 8 to 24 different words drawn from 31, then the same words again. For each
// seed: the steps at which the second copy, from its second word (nothing before tells the first), first passes 20%,
// 50% and 80% right (measured every 10 steps on 60 test sequences), and at the end each head's average weight, in the second copy,
// on the token just before and on the token after the earlier copy. The page runs the same code (toySeq and the worker
// in gpt/index.html).
// node toy.js [seeds=8] [steps=2000] [fixed=1: the repeat always 16 long, where a head can just read 15 back]
const L = require("./lib.js"), G = L.GPT;
const a = L.args({ seeds: 8, first: 1, steps: 2000, fixed: 0 });
const SHAPE = { V: 32, T: 49, d: 32, L: 2, H: 2, M: 0 }, KMIN = a.fixed ? 16 : 8, KMAX = a.fixed ? 16 : 24;
function seq(r) {
  const k = KMIN + Math.floor(r() * (KMAX - KMIN + 1)), pool = Array.from({ length: SHAPE.V - 1 }, (_, i) => i + 1), s = [];
  for (let j = 0; j < k; j++) s.push(pool.splice(Math.floor(r() * pool.length), 1)[0]);
  const t = Uint16Array.from([0, ...s, ...s]); return { tokens: t.subarray(0, t.length - 1), next: t.subarray(1), k };
}
const tr = L.mulberry32(99), test = Array.from({ length: 60 }, () => seq(tr));
function measure(m) {
  let right = 0, n = 0; const prev = new Float64Array(4), ind = new Float64Array(4); let pos = 0;
  for (const { tokens, next, k } of test) {
    const f = G.forward(m, tokens);
    for (let i = k + 1; i < next.length; i++) { right += L.argmax(f.p, i * SHAPE.V, SHAPE.V) === next[i]; n++; }
    for (let i = k + 1; i < tokens.length; i++) { pos++; for (let l = 0; l < 2; l++) for (let h = 0; h < 2; h++) { const b = (i * 2 + h) * SHAPE.T; prev[l * 2 + h] += f.blocks[l].w[b + 1]; ind[l * 2 + h] += f.blocks[l].w[b + k - 1]; } }
  }
  return { acc: right / n, prev: Array.from(prev, (x) => x / pos), ind: Array.from(ind, (x) => x / pos) };
}
const pc = (x) => (100 * x).toFixed(0) + "%";
for (let seed = a.first; seed < a.first + a.seeds; seed++) {
  const r = L.mulberry32(seed), m = G.makeModel(SHAPE, r), adam = G.makeAdam(m), g = G.zeros(m), one = G.zeros(m);
  let at = -1, a20 = -1, a80 = -1;
  for (let s = 1; s <= a.steps; s++) {
    G.trainStep(m, Array.from({ length: 16 }, () => seq(r)), 0.01, adam, g, one);
    if (s % 10 === 0 && a80 < 0) { const acc = measure(m).acc; if (a20 < 0 && acc > 0.2) a20 = s; if (at < 0 && acc > 0.5) at = s; if (acc > 0.8) a80 = s; }
  }
  const x = measure(m);
  console.log(`seed ${seed}: second copy past 20% at step ${a20}, 50% at ${at}, 80% at ${a80}; at the end ${pc(x.acc)} right | before / after the copy: ${[0, 1, 2, 3].map((k) => `block ${(k >> 1) + 1} head ${(k & 1) + 1} ${pc(x.prev[k])} / ${pc(x.ind[k])}`).join(", ")}`);
}
