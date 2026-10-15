// Check the page's Python translation against its JavaScript: the same probabilities and attention weights, and the
// same losses and weights after three Adam steps from the same start, on short pieces of real stories with the real
// shapes (a window of 12, so that some positions cannot see the story's start). Needs python3; no packages.
const L = require("./lib.js"), fs = require("fs"), path = require("path"), { execFileSync } = require("child_process");
const dir = L.work("py"); fs.mkdirSync(dir, { recursive: true });
fs.writeFileSync(path.join(dir, "gpt.py"), L.script("gpt-py") + "\n");
const cfg = { V: 2000, T: 12, d: 64, L: 4, H: 4, M: 256 };
const r = L.mulberry32(7), m = L.GPT.makeModel(cfg, r), adam = L.GPT.makeAdam(m);
// weights that are not all at their starting values: distance scores, gains, shifts and biases
for (const key of ["R", "g1", "s1", "g2", "s2", "b1", "b2", "gf", "sf", "bout"]) for (let i = 0; i < m[key].length; i++) m[key][i] += 0.3 * Math.sin(i + key.length);
const fresh = L.GPT.makeModel(cfg, L.mulberry32(7)); // the starting weights as makeModel draws them, before the changes above
const flat = (mm) => Object.fromEntries(L.GPT.KEYS.map((k) => [k, Array.from(mm[k])]));
const { stories } = L.storiesOf("train.bin");
const piece = (s, n) => ({ tokens: Array.from(stories[s].tokens.subarray(0, n)), next: Array.from(stories[s].next.subarray(0, n)) });
const batches = [[piece(10, 20), piece(11, 15)], [piece(12, 18)], [piece(13, 22), piece(14, 9)]];
const init = flat(m), f0 = L.GPT.forward(m, Uint16Array.from(batches[0][0].tokens), 5);
const p0 = Array.from(f0.p.slice(0, 40)), w0 = Array.from(f0.blocks[2].w.slice(0, 15 * cfg.H * cfg.T)), x0 = Array.from(f0.xs[4].slice(0, 15 * cfg.d));
const losses = batches.map((b) => L.GPT.trainStep(m, b.map((s) => ({ tokens: Uint16Array.from(s.tokens), next: s.next })), 0.01, adam));
fs.writeFileSync(path.join(dir, "fix.json"), JSON.stringify({ cfg, fresh: flat(fresh), init, batches: batches.map((b) => b.map((s) => [s.tokens, s.next])), p0, w0, x0, losses, after: flat(m) }));
console.log(execFileSync("python3", ["-I", path.join(__dirname, "pycheck.py"), dir], { encoding: "utf8" }));
