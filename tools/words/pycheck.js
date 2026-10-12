// Check the page's Python translation against its JavaScript: the same tidying, tokens and learned merges,
// and the same losses and weights after three Adam steps from the same start. Writes the Python code and test
// cases to work/py/ and runs pycheck.py on them (needs python3; no packages).
const L = require("./lib.js"), fs = require("fs"), path = require("path"), { execFileSync } = require("child_process");
const dir = L.work("py"); fs.mkdirSync(dir, { recursive: true });
fs.writeFileSync(path.join(dir, "tok.py"), L.script("tok-py") + "\n"); fs.writeFileSync(path.join(dir, "nwm.py"), L.script("nwm-py") + "\n");
const tk = L.tokenizer(), raw = L.stories(L.work("dl", "valid.txt"), 300).concat(["Hello -- world... “Quoted” and ‘single’ — café 911 don't kids' toys. (Paren)"]);
const counts = new Map(); for (const s of raw.slice(0, 150)) for (const p of L.TOK.pieces(s)) if (/^[a-z]/.test(p)) counts.set(p, (counts.get(p) || 0) + 1);
const r = L.mulberry32(7), m = L.NWM.makeModel({ V: 2000, k: 8, d: 32, h: 128 }, r), adam = L.NWM.makeAdam(m);
const init = Object.fromEntries(L.NWM.KEYS.map((k) => [k, Array.from(m[k])]));
const tok = L.loadTokens("train.bin"), st = L.starts(tok), batches = [];
for (let s = 0; s < 3; s++) { const b = []; for (let i = 0; i < 3; i++) { const t = 1000 + 7919 * (s * 3 + i); b.push({ context: L.ctxAt(tok, st, t, 8), next: tok[t] }); } batches.push(b); }
const losses = batches.map((b) => L.NWM.trainStep(m, b, 0.01, adam));
fs.writeFileSync(path.join(dir, "fix.json"), JSON.stringify({
  raw, tidy: raw.map(L.TOK.tidy), ids: raw.map((s) => tk.encode(s)), decoded: raw.map((s) => tk.decode(tk.encode(s))), merges: L.merges(), counts: [...counts], small: L.TOK.learnMerges(counts, 300),
  init, batches: batches.map((b) => b.map((e) => [e.context, e.next])), losses, after: Object.fromEntries(L.NWM.KEYS.map((k) => [k, Array.from(m[k])])), p0: Array.from(L.NWM.forward(m, batches[0][0].context).p.slice(0, 50)),
}));
console.log(execFileSync("python3", ["-I", path.join(__dirname, "pycheck.py"), dir], { encoding: "utf8" }));
