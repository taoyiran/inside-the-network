// Shared helpers for chapter 8's build scripts. The tokenizer, chapter 6's network and the token files come from
// chapter 6 (../words/lib.js and ../words/work/), chapter 7's network from ../attention/lib.js; the GPT is read from
// gpt/index.html itself (<script id="gpt">), so the scripts always run the page's own code.
// Checkpoints and logs live in tools/gpt/work/, which git ignores.
const fs = require("fs"), path = require("path");
const A7 = require("../attention/lib.js");
const WORK = path.join(__dirname, "work");
const work = (...p) => path.join(WORK, ...p);
const PAGE = path.join(__dirname, "..", "..", "gpt", "index.html"), html = fs.readFileSync(PAGE, "utf8");
const script = (id) => { const m = html.match(new RegExp(`<script(?: type="text/plain")? id="${id}">\\n([\\s\\S]*?)\\n</script>`)); if (!m) throw new Error("no script " + id); return m[1]; };
const EXPORTS = "\nreturn { makeModel, blockOf, mul, mulBack, norm, normBack, gelu, geluSlope, forward, backward, zeros, makeAdam, adamStep, trainStep, KEYS, PER_BLOCK };";
// the page's code, or a variant of it made by replacing parts of its text (each replacement must apply exactly once)
function codeWith(edits = []) {
  let src = script("gpt");
  for (const [a, b] of edits) { const n = src.split(a).length - 1; if (n !== 1) throw new Error(`variant: "${a.slice(0, 60)}" found ${n} times`); src = src.replace(a, b); }
  return new Function(src + EXPORTS)();
}
const GPT = codeWith();
// The variants for the comparison in the chapter: the same network without its residual connections (a block's
// output is only what its parts make, not added to the row), or without layer norms (each norm passes its row on).
const VARIANTS = {
  plain: [],
  nores: [
    ["mid[j] = x[j] + att[j];", "mid[j] = att[j];"],
    ["out[j] = mid[j] + add[j];", "out[j] = add[j];"],
    ["const dmid = Float64Array.from(dx),", "const dmid = new Float64Array(n * d),"],
    ["const dxl = Float64Array.from(dmid),", "const dxl = new Float64Array(n * d),"],
  ],
  // the same without residual connections, and without the smaller start of the tables whose results are added to
  // the rows (GPT-2's 1/√(2L), which suits a network with residual connections)
  nores1: [
    ["const ones = (n) => new Float64Array(n).fill(1), deep = 1 / Math.sqrt(2 * L);", "const ones = (n) => new Float64Array(n).fill(1), deep = 1;"],
    ["mid[j] = x[j] + att[j];", "mid[j] = att[j];"],
    ["out[j] = mid[j] + add[j];", "out[j] = add[j];"],
    ["const dmid = Float64Array.from(dx),", "const dmid = new Float64Array(n * d),"],
    ["const dxl = Float64Array.from(dmid),", "const dxl = new Float64Array(n * d),"],
  ],
  nonorm: [
    ["function norm(xs, n, d, g, s) {", "function norm(xs, n, d, g, s) { return { y: Float64Array.from(xs.subarray(0, n * d)) };"],
    ["function normBack(dy, nm, n, d, g, gg, gs, dx) {", "function normBack(dy, nm, n, d, g, gg, gs, dx) { for (let j = 0; j < n * d; j++) dx[j] += dy[j]; return;"],
  ],
};
const variant = (name) => { if (name && !VARIANTS[name]) throw new Error(`no variant "${name}" (${Object.keys(VARIANTS).join(", ")})`); return name && name !== "plain" ? codeWith(VARIANTS[name]) : GPT; };
// checkpoints: a 128-byte JSON header with the shape, then the KEYS in order as 64-bit floats
const sizes = ({ V, T, d, L, H, M }) => ({ E: V * d, g1: L * d, s1: L * d, Wq: L * d * d, Wk: L * d * d, Wv: L * d * d, R: L * H * T, Wo: L * d * d, g2: L * d, s2: L * d, W1: L * M * d, b1: L * M, W2: L * d * M, b2: L * d, gf: d, sf: d, Wout: V * d, bout: V });
const shapeOf = (m) => ({ V: m.V, T: m.T, d: m.d, L: m.L, H: m.H, M: m.M });
function saveModel(m, name) {
  fs.mkdirSync(work("models"), { recursive: true });
  const head = Buffer.from(JSON.stringify(shapeOf(m)).padEnd(128));
  fs.writeFileSync(work("models", name), Buffer.concat([head, ...GPT.KEYS.map((key) => Buffer.from(m[key].buffer, m[key].byteOffset, m[key].byteLength))]));
}
function loadModel(name) {
  const b = fs.readFileSync(name.includes("/") ? name : work("models", name)), cfg = JSON.parse(b.slice(0, 128).toString()), m = { ...cfg }, n = sizes(cfg);
  let o = 128;
  for (const key of GPT.KEYS) { m[key] = new Float64Array(b.buffer.slice(b.byteOffset + o, b.byteOffset + o + n[key] * 8)); o += n[key] * 8; }
  return m;
}
const count = (m) => GPT.KEYS.reduce((a, key) => a + m[key].length, 0);
// loss and right first guesses over whole stories (chapter 7's loop, with the GPT's forward pass)
const evaluate = (m, stories, code = GPT) => A7.evaluate(m, stories, code);
// The page's own measures (gpt/index.html), for checkpoints. Invented names: each name story (a test story among the
// first 100 whose hero, by rename.js's rule, comes back from further than 8 tokens) gets one of the 6 names in turn; at every
// mention, each piece after the first is guessed at the position before it, split by whether that pair of pieces
// ever appears in the training stories. Repeats: 50 sequences of <end>, 40 tokens drawn from the 1,000 most common,
// then the same 40 again (mulberry32(7)); right first guesses in the second copy, and each head's weights there.
// the 6 invented names and the rules for names (which mentions count, who is a story's hero) are rename.js's
const RN = require("./rename.js"), INVENT = RN.INVENT, hero = RN.hero;
// the story with every mention of its hero X as a word replaced by the pieces Z (rename.js's swapName, as the page
// does it); where each mention's pieces start
function invented(tokens, X, Z) {
  const { out, at } = RN.swapName(tokens, X, Z, RN.kit());
  return { toks: Uint16Array.from(out), ment: at };
}
let PM = null;
function pageSetup() {
  if (PM) return PM;
  const tk = A7.tokenizer(), tok = A7.loadTokens("train.bin"), test = A7.storiesOf("test.bin").stories.slice(0, 100);
  const pairs = new Map(); for (const w of INVENT) { const Z = tk.encode(w); for (let j = 1; j < Z.length; j++) pairs.set(Z[j - 1] * 2000 + Z[j], 0); }
  for (let t = 0; t + 1 < tok.length; t++) { const k = tok[t] * 2000 + tok[t + 1]; if (pairs.has(k)) pairs.set(k, pairs.get(k) + 1); }
  const list = []; test.forEach(({ tokens, next }) => { const X = hero(tokens); if (X >= 0 && A7.farPositions(tokens, next, X).length) list.push({ tokens, X }); });
  const items = list.map(({ tokens, X }, i) => {
    const Z = tk.encode(INVENT[i % INVENT.length]), { toks, ment } = invented(tokens, X, Z);
    const at = []; ment.forEach((m, k) => { for (let j = 1; j < Z.length; j++) at.push({ r: m + j - 1, want: Z[j], first: k === 0, seen: pairs.get(Z[j - 1] * 2000 + Z[j]) > 0 }); });
    return { toks, at, Z };
  });
  const cnt = new Uint32Array(2000); for (const t of tok) cnt[t]++;
  const common = Array.from(cnt.keys()).filter((t) => t > 0).sort((a, b) => cnt[b] - cnt[a] || a - b).slice(0, 1000), r = A7.mulberry32(7);
  const reps = Array.from({ length: 50 }, () => { const q = Array.from({ length: 40 }, () => common[Math.floor(r() * common.length)]); return Uint16Array.from([0, ...q, ...q]); });
  return (PM = { items, reps, stories: list.length });
}
function pageMeasures(m, code = GPT) {
  // c[kind]: [right at first mentions, first mentions, right at later ones, later ones]; wrong: at later mentions of
  // never-seen pairs, how often the first guess is a piece of the same name, but the wrong one
  const { items, reps } = pageSetup(), c = { seen: [0, 0, 0, 0], unseen: [0, 0, 0, 0], all: [0, 0, 0, 0] }; let wrong = 0;
  for (const { toks, at, Z } of items) {
    const { p } = code.forward(m, toks);
    for (const { r, want, first, seen } of at) {
      const g = argmax(p, r * m.V, m.V), ok = g === want ? 1 : 0, o = first ? 0 : 2;
      for (const k of [seen ? "seen" : "unseen", "all"]) { c[k][o] += ok; c[k][o + 1]++; }
      if (!first && !seen && !ok && Z.includes(g)) wrong++;
    }
  }
  const { L = 0, H, T } = m, n = 40, prev = new Float64Array(L * H), ind = new Float64Array(L * H); let right = 0, pos = 0;
  for (const s of reps) {
    const f = code.forward(m, s);
    for (let i = n + 1; i < s.length - 1; i++) if (argmax(f.p, i * m.V, m.V) === s[i + 1]) right++; // the second copy's tokens after its first
    if (f.blocks) for (let i = n + 2; i < s.length; i++) { pos++; for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) { const w = f.blocks[l].w, b = (i * H + h) * T; prev[l * H + h] += w[b + 1]; ind[l * H + h] += w[b + n - 1]; } }
  }
  const rate = ([f, fn, l, ln]) => ({ first: f / Math.max(1, fn), later: l / Math.max(1, ln), nFirst: fn, nLater: ln });
  return { seen: rate(c.seen), unseen: rate(c.unseen), all: rate(c.all), wrongPiece: wrong / Math.max(1, c.unseen[3]), second: right / (reps.length * (n - 1)), prev: Array.from(prev, (x) => x / Math.max(1, pos)), ind: Array.from(ind, (x) => x / Math.max(1, pos)) };
}
// how much gradient reaches each block's weights: the size (root of the sum of squares) of the gradient of each block's
// weights, summed over 16 training stories (101 to 116), to 4 significant digits; code is the GPT or a variant
let GS = null;
function blockGrads(code, m) {
  const g = code.zeros(m); GS = GS || A7.storiesOf("train.bin").stories.slice(100, 116);
  for (const st of GS) code.backward(m, st.tokens, st.next, code.forward(m, st.tokens), g);
  return Array.from({ length: m.L }, (_, l) => { const B = code.blockOf(g, l); let ss = 0; for (const key of code.PER_BLOCK) for (const x of B[key]) ss += x * x; return +Math.sqrt(ss).toPrecision(4); });
}
const argmax = A7.argmax;
module.exports = { INVENT, RN, hero, invented, pageSetup, pageMeasures, blockGrads, ...A7, A7, ATN: A7.ATN, loadModel7: A7.loadModel, PAGE, html, script, work, GPT, codeWith, VARIANTS, variant, sizes, shapeOf, saveModel, loadModel, count, evaluate };
