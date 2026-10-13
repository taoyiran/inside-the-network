// Secret 4, piece by piece: each of the page's 6 invented names is given to the hero of every one of the page's name
// stories (test stories whose name comes back from further than 8 tokens). For every piece after the first, at the
// first mention and at later mentions: how often it is the first guess, and how often that pair of pieces appears in
// the training stories. node pieces.js F8_19000.bin
const L = require("./lib.js");
const tk = L.tokenizer(), V = tk.vocab, m = L.loadModel(process.argv[2]), tok = L.loadTokens("train.bin");
const list = [];
L.storiesOf("test.bin").stories.slice(0, 100).forEach(({ tokens, next }) => { const X = L.nameOf(tokens); if (X >= 0 && L.farPositions(tokens, next, X).length) list.push({ tokens, X }); });
const pair = (a, b) => { let n = 0; for (let t = 0; t + 1 < tok.length; t++) if (tok[t] === a && tok[t + 1] === b) n++; return n; };
console.log(`${list.length} name stories`);
for (const w of ["zarko", "vorna", "quilla", "morvin", "dashu", "tilda"]) {
  const Z = tk.encode(w), hit = Z.map(() => ({ f: 0, fn: 0, l: 0, ln: 0 }));
  for (const { tokens, X } of list) {
    const out = [0], ment = []; for (let j = 1; j < tokens.length; j++) if (tokens[j] === X) { ment.push(out.length); out.push(...Z); } else out.push(tokens[j]);
    const f = L.ATN.forward(m, Uint16Array.from(out));
    ment.forEach((mm, k) => { for (let j = 1; j < Z.length; j++) { const ok = L.argmax(f.p, (mm + j - 1) * m.V, m.V) === Z[j], h = hit[j]; if (k === 0) { h.fn++; h.f += ok; } else { h.ln++; h.l += ok; } } });
  }
  console.log(w.padEnd(7), Z.slice(1).map((z, i) => `“${V[Z[i]]}” → “${V[z]}” (${pair(Z[i], z)}× in training): first ${hit[i + 1].f}/${hit[i + 1].fn}, later ${hit[i + 1].l}/${hit[i + 1].ln}`).join(" | "));
}
