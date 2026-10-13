// The backward pass against central differences, on a small network and a short story, for every weight array,
// with and without `from` (outputs only for the later positions), and with a window shorter than the story.
const L = require("./lib.js");
const rand = L.mulberry32(5);
let worst = 0;
for (const [T, from] of [[16, 0], [5, 0], [5, 7]]) {
  const m = L.ATN.makeModel({ V: 30, T, d: 6, H: 3, dk: 4, dv: 5, h: 7 }, rand, 0.5);
  for (const key of ["R", "b1", "b2"]) for (let i = 0; i < m[key].length; i++) m[key][i] = 0.3 * (rand() - 0.5);
  const tokens = Array.from({ length: 12 }, (_, i) => (i ? Math.floor(rand() * 30) : 0)), next = Array.from({ length: 12 - from }, () => Math.floor(rand() * 30));
  tokens[9] = tokens[3]; // a token read twice
  const loss = () => { const { p } = L.ATN.forward(m, tokens, from); let s = 0; for (let r = 0; r < next.length; r++) s -= Math.log(p[r * m.V + next[r]]); return s; };
  const g = L.ATN.zeros(m);
  L.ATN.backward(m, tokens, next, L.ATN.forward(m, tokens, from), g);
  const line = [];
  for (const key of L.ATN.KEYS) {
    let kw = 0, checked = 0;
    for (let i = 0; i < m[key].length; i++) {
      const old = m[key][i], e = 1e-5;
      m[key][i] = old + e; const up = loss(); m[key][i] = old - e; const dn = loss(); m[key][i] = old;
      const num = (up - dn) / (2 * e), ana = g[key][i];
      if (Math.abs(num) < 1e-6 && Math.abs(ana) < 1e-6) continue;
      kw = Math.max(kw, Math.abs(num - ana) / Math.max(1e-4, Math.abs(num) + Math.abs(ana))); checked++;
    }
    worst = Math.max(worst, kw); line.push(`${key} ${checked} ${kw.toExponential(1)}`);
  }
  console.log(`T=${T} from=${from}:`, line.join(" | "));
}
console.log("largest relative difference", worst.toExponential(2));
