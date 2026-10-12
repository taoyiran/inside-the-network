// Shared helpers for chapter 6's build scripts. The tokenizer and the network are pulled out of
// words/index.html itself (<script id="tok"> and <script id="nwm">), so the scripts always run the page's own code.
// Downloads, token files, checkpoints and logs live in tools/words/work/, which git ignores.
const fs = require("fs"), path = require("path");
const PAGE = path.join(__dirname, "..", "..", "words", "index.html");
const WORK = path.join(__dirname, "work");
const html = fs.readFileSync(PAGE, "utf8");
const script = (id) => { const m = html.match(new RegExp(`<script(?: type="text/plain")? id="${id}">\\n([\\s\\S]*?)\\n</script>`)); if (!m) throw new Error("no script " + id); return m[1]; };
const TOK = new Function(script("tok") + "\nreturn { tidy, pieces, letters, glue, learnMerges, cutWord, makeTokenizer, END };")();
const NWM = new Function(script("nwm") + "\nreturn { makeModel, forward, backward, zeros, makeAdam, trainStep, KEYS };")();
const work = (...p) => path.join(WORK, ...p);
// the stories of a TinyStories text file, in order, without the text before the first <|endoftext|>
// (in the validation file, the end of a cut-off story) or after the last one
const stories = (file, n) => fs.readFileSync(file, "utf8").split("<|endoftext|>").slice(1, -1).map((s) => s.trim()).filter(Boolean).slice(0, n);
function loadTokens(name) { const b = fs.readFileSync(work(name)); return new Uint16Array(b.buffer, b.byteOffset, b.length / 2); }
// start[t]: where the story holding position t begins (the <end> token closes a story)
function starts(tok) { const s = new Int32Array(tok.length); let st = 0; for (let t = 0; t < tok.length; t++) { s[t] = st; if (tok[t] === 0) st = t + 1; } return s; }
// the k tokens before position t, with 0 (<end>) before the story's start: the same contexts as the page's contextAt
function ctxAt(tok, st, t, k) { const c = new Array(k); for (let j = 0; j < k; j++) { const i = t - k + j; c[j] = i >= st[t] ? tok[i] : 0; } return c; }
function mulberry32(a) { return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
// checkpoints: a 64-byte JSON header with the shape, then E, W1, b1, W2, b2 as 64-bit floats
function saveModel(m, name) { fs.mkdirSync(work("models"), { recursive: true }); const head = Buffer.from(JSON.stringify({ V: m.V, k: m.k, d: m.d, h: m.h }).padEnd(64)); fs.writeFileSync(work("models", name), Buffer.concat([head, ...NWM.KEYS.map((key) => Buffer.from(m[key].buffer, m[key].byteOffset, m[key].byteLength))])); }
function loadModel(name) { const b = fs.readFileSync(work("models", name)), cfg = JSON.parse(b.slice(0, 64).toString()), m = { ...cfg }; let o = 64; const n = { E: cfg.V * cfg.d, W1: cfg.h * cfg.k * cfg.d, b1: cfg.h, W2: cfg.V * cfg.h, b2: cfg.V }; for (const key of NWM.KEYS) { m[key] = new Float64Array(b.buffer.slice(b.byteOffset + o, b.byteOffset + o + n[key] * 8)); o += n[key] * 8; } return m; }
const merges = () => JSON.parse(fs.readFileSync(work("merges.json"), "utf8"));
const tokenizer = () => TOK.makeTokenizer(merges());
// accuracy and loss of a model on given positions of a token stream
function evaluate(m, tok, st, positions) {
  let loss = 0, right = 0;
  for (const t of positions) { const { p } = NWM.forward(m, ctxAt(tok, st, t, m.k)); loss -= Math.log(p[tok[t]]); let b = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i; if (b === tok[t]) right++; }
  return { loss: loss / positions.length, acc: right / positions.length };
}
// numbered arguments like lr=0.001 h=128
const args = (defaults) => ({ ...defaults, ...Object.fromEntries(process.argv.slice(2).filter((a) => a.includes("=")).map((a) => a.split("=")).map(([k, v]) => [k, isNaN(+v) ? v : +v])) });
module.exports = { PAGE, WORK, html, script, TOK, NWM, work, stories, loadTokens, starts, ctxAt, mulberry32, saveModel, loadModel, merges, tokenizer, evaluate, args };
