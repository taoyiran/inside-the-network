// Check the page's Python translation against its JavaScript: the same probabilities and attention weights, and the
// same losses and weights after three Adam steps from the same start, on short pieces of real stories with the real
// shapes (a window of 12, so that some positions cannot see the story's start). Needs python3; no packages.
const L = require("./lib.js"), fs = require("fs"), path = require("path"), { execFileSync } = require("child_process");
const dir = L.work("py"); fs.mkdirSync(dir, { recursive: true });
fs.writeFileSync(path.join(dir, "attn.py"), L.script("attn-py") + "\n");
const cfg = { V: 2000, T: 12, d: 32, H: 8, dk: 8, dv: 16, h: 128 };
const r = L.mulberry32(7), m = L.ATN.makeModel(cfg, r), adam = L.ATN.makeAdam(m);
for (let i = 0; i < m.R.length; i++) m.R[i] = 0.5 * Math.sin(i); // distance scores that are not all 0
const init = Object.fromEntries(L.ATN.KEYS.map((k) => [k, Array.from(m[k])]));
const { stories } = L.storiesOf("train.bin");
const piece = (s, n) => ({ tokens: Array.from(stories[s].tokens.subarray(0, n)), next: Array.from(stories[s].next.subarray(0, n)) });
const batches = [[piece(10, 20), piece(11, 15)], [piece(12, 18)], [piece(13, 22), piece(14, 9)]];
const f0 = L.ATN.forward(m, batches[0][0].tokens, 5);
const p0 = Array.from(f0.p.slice(0, 40)), w0 = Array.from(f0.w.slice(0, 3 * cfg.H * cfg.T));
const losses = batches.map((b) => L.ATN.trainStep(m, b, 0.01, adam, L.ATN.zeros(m)));
fs.writeFileSync(path.join(dir, "fix.json"), JSON.stringify({ cfg, init, batches: batches.map((b) => b.map((s) => [s.tokens, s.next])), p0, w0, losses, after: Object.fromEntries(L.ATN.KEYS.map((k) => [k, Array.from(m[k])])) }));
console.log(execFileSync("python3", ["-I", path.join(__dirname, "pycheck.py"), dir], { encoding: "utf8" }));
