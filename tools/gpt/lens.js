// Secret 4, the logit lens: each position's row read after each block (0: the token's own row) through the last layer
// norm and the output layer, as if it were the last; how often that first guess is right, over the page's 100 test
// stories (NSTORIES=1000: all the test stories), and the examples the page shows.
// node lens.js G_9500.bin
const L = require("./lib.js");
const m = L.loadModel(process.argv[2]), G = L.GPT, NS = +(process.env.NSTORIES || 100), test = L.storiesOf("test.bin").stories.slice(0, NS);
const right = new Float64Array(m.L + 1); let n = 0;
for (const { tokens, next } of test) {
  const f = G.forward(m, tokens);
  for (let l = 0; l <= m.L; l++) { const nf = G.norm(f.xs[l], tokens.length, m.d, m.gf, m.sf), s = G.mul(m.Wout, m.bout, nf.y, tokens.length, m.V, m.d); for (let r = 0; r < next.length; r++) if (L.argmax(s, r * m.V, m.V) === next[r]) right[l]++; }
  n += next.length;
}
console.log(`${n} tokens; right first guess read after each block:`, Array.from(right, (x, l) => `${l ? "block " + l : "own row"} ${(100 * x / n).toFixed(1)}%`).join(", "));
const tk = L.tokenizer();
for (const text of ["once upon a", "once upon a time, there was a little", "lily and ben went to the park. lily"]) {
  const toks = Uint16Array.from([0, ...tk.encode(text)]), f = G.forward(m, toks, toks.length - 1), i = toks.length - 1;
  console.log(`“${text}”:`, Array.from({ length: m.L + 1 }, (_, l) => { const nf = G.norm(f.xs[l].subarray(i * m.d), 1, m.d, m.gf, m.sf), s = G.mul(m.Wout, m.bout, nf.y, 1, m.V, m.d), t = L.argmax(s); return `${l}: ${tk.vocab[t]}`; }).join(" | "));
}
