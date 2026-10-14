// Shared helpers for chapter 7's build scripts. The tokenizer, chapter 6's network and the token files come from
// chapter 6 (../words/lib.js and ../words/work/); the attention network is read from attention/index.html itself
// (<script id="attn">), so the scripts always run the page's own code.
// Checkpoints and logs live in tools/attention/work/, which git ignores.
const fs = require("fs"), path = require("path");
const W6 = require("../words/lib.js");
const WORK = path.join(__dirname, "work");
const work = (...p) => path.join(WORK, ...p);
const PAGE = path.join(__dirname, "..", "..", "attention", "index.html"), html = fs.readFileSync(PAGE, "utf8");
const script = (id) => { const m = html.match(new RegExp(`<script(?: type="text/plain")? id="${id}">\\n([\\s\\S]*?)\\n</script>`)); if (!m) throw new Error("no script " + id); return m[1]; };
const ATN = new Function(script("attn") + "\nreturn { makeModel, project, forward, backward, zeros, makeAdam, adamStep, trainStep, KEYS };")();
// The stories of a token file as { tokens, next }: tokens = <end>, then the story; next = the story, then <end>.
// In the token files each story is followed by 0 (<end>), so story s is tok[a - 1 … b] with the 0 before it
// (the first story gets one in front). starts[s]: where story s's own tokens begin in the file.
function storiesOf(name) {
  const tok = W6.loadTokens(name), pad = new Uint16Array(tok.length + 1);
  pad.set(tok, 1);
  const out = [], starts = [];
  let a = 0;
  for (let t = 0; t < tok.length; t++) if (tok[t] === 0) { out.push({ tokens: pad.subarray(a, t + 1), next: pad.subarray(a + 1, t + 2) }); starts.push(a); a = t + 1; }
  return { tok, stories: out, starts };
}
// checkpoints: a 128-byte JSON header with the shape, then the KEYS in order as 64-bit floats
function sizes(cfg) { const { V, T, d, H, dk, dv, h } = cfg, D = d + H * dv; return { E: V * d, Wq: H * dk * d, Wk: H * dk * d, Wv: H * dv * d, R: H * T, W1: h * D, b1: h, W2: V * h, b2: V }; }
function saveModel(m, name) {
  fs.mkdirSync(work("models"), { recursive: true });
  const { V, T, d, H, dk, dv, h } = m, head = Buffer.from(JSON.stringify({ V, T, d, H, dk, dv, h }).padEnd(128));
  fs.writeFileSync(work("models", name), Buffer.concat([head, ...ATN.KEYS.map((key) => Buffer.from(m[key].buffer, m[key].byteOffset, m[key].byteLength))]));
}
function loadModel(name) {
  const b = fs.readFileSync(work("models", name)), cfg = JSON.parse(b.slice(0, 128).toString()), m = { ...cfg }, n = sizes(cfg);
  let o = 128;
  for (const key of ATN.KEYS) { m[key] = new Float64Array(b.buffer.slice(b.byteOffset + o, b.byteOffset + o + n[key] * 8)); o += n[key] * 8; }
  return m;
}
// the shipped attention/weights.js (window.ITN_ATTN), and the network in it as the page has it: 32-bit floats
function shippedFile() { const ctx = {}; new Function("window", fs.readFileSync(path.join(__dirname, "..", "..", "attention", "weights.js"), "utf8"))(ctx); return ctx.ITN_ATTN; }
function loadShipped() {
  const W = shippedFile(), b = Buffer.from(W.net, "base64"), m = { ...W.shape }, n = sizes(W.shape); let o = 0;
  for (const key of ATN.KEYS) { m[key] = new Float64Array(n[key]); for (let i = 0; i < n[key]; i++, o += 4) m[key][i] = b.readFloatLE(o); }
  return m;
}
// loss and right first guesses over whole stories
function evaluate(m, stories) {
  let loss = 0, right = 0, n = 0;
  for (const { tokens, next } of stories) {
    const { p } = ATN.forward(m, tokens);
    for (let r = 0; r < next.length; r++) {
      loss -= Math.log(p[r * m.V + next[r]]);
      let b = 0; for (let i = 1; i < m.V; i++) if (p[r * m.V + i] > p[r * m.V + b]) b = i;
      if (b === next[r]) right++; n++;
    }
  }
  return { loss: loss / n, acc: right / n, n };
}
// the story picked by training token t: the last story that starts at or before t
function storyAt(starts, t) { let lo = 0, hi = starts.length - 1; while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (starts[mid] <= t) lo = mid; else hi = mid - 1; } return lo; }
const argmax = (p, o = 0, n = p.length - o) => { let b = 0; for (let i = 1; i < n; i++) if (p[o + i] > p[o + b]) b = i; return b; };
// The page's names: one-token names from the training stories (each follows "named" at least 20 times there); a
// story's name is the one mentioned most (ties: the lower token number); farPositions are the positions r (reading
// tokens[r], guessing next[r]) whose next token is that name and whose last mention is more than 8 tokens back.
const NAME_WORDS = "tim lily tom sue max lucy mia amy sam bob ben jane jack anna sarah mark emma fred lisa leo joe john ann sally kitty".split(" ");
let NAMES_ = null;
const names = () => NAMES_ || (NAMES_ = NAME_WORDS.map((w) => W6.tokenizer().id.get(w)).filter((x) => x !== undefined));
function nameOf(tokens) {
  const ids = names(), cnt = new Map(); for (const x of tokens) if (ids.includes(x)) cnt.set(x, (cnt.get(x) || 0) + 1);
  return cnt.size ? [...cnt].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0] : -1;
}
function farPositions(tokens, next, X, K = 8) {
  const at = [];
  for (let r = 0; r < next.length; r++) if (next[r] === X) { let last = -1; for (let i = r; i >= 1; i--) if (tokens[i] === X) { last = i; break; } if (last >= 1 && r - last >= K) at.push(r); }
  return at;
}
// where each head's weight goes over some stories, as the page's score() measures it: the share of all weight in
// each distance bin, and the 4 tokens that get the most weight per appearance (among tokens read at least 5 times)
const NB = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 7], [8, 15], [16, 63], [64, 255]];
function headStats(m, stories) {
  const { V, T, H } = m, binOf = new Uint8Array(T); NB.forEach(([a, b], k) => { for (let t = a; t <= Math.min(b, T - 1); t++) binOf[t] = k; });
  const bins = new Float64Array(H * NB.length), reads = new Float64Array(H * V), seen = new Float64Array(V); let n = 0;
  for (const { tokens, next } of stories) {
    const f = ATN.forward(m, tokens);
    for (let r = 0; r < next.length; r++) { n++; seen[tokens[r]]++; for (let h = 0; h < H; h++) for (let t = 0; t <= Math.min(r, T - 1); t++) { const w = f.w[(r * H + h) * T + t]; bins[h * NB.length + binOf[t]] += w; reads[h * V + tokens[r - t]] += w; } }
  }
  const tops = [];
  for (let h = 0; h < H; h++) { const per = []; for (let t = 0; t < V; t++) if (seen[t] >= 5) per.push([t, reads[h * V + t] / seen[t]]); per.sort((a, b) => b[1] - a[1]); tops.push(...per.slice(0, 4).map(([t]) => t)); }
  return { bins: Array.from(bins, (v) => +(v / n).toFixed(4)), tops, reads, seen, n };
}
// a run's seed, from the settings line its log starts with
const seedOf = (run) => JSON.parse(fs.readFileSync(work("runs", run + ".txt"), "utf8").split("\n")[0]).seed;
module.exports = { ...W6, W6, WORK6: W6.WORK, PAGE, html, script, work, storyAt, argmax, NAME_WORDS, names, nameOf, farPositions, NB, headStats, seedOf, ATN, storiesOf, sizes, saveModel, loadModel, shippedFile, loadShipped, evaluate };
