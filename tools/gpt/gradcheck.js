// The backward pass against central differences, on small networks: every weight array, a story longer than the
// window, outputs from a later position (`from`), and the variants without residual connections or layer norms.
// node gradcheck.js
const L = require("./lib.js");
let worst = 0;
for (const [vname, cfg, n, from] of [["plain", { V: 23, T: 5, d: 8, L: 2, H: 2, M: 12 }, 9, 0], ["plain", { V: 23, T: 16, d: 12, L: 3, H: 3, M: 10 }, 11, 6], ["plain", { V: 17, T: 4, d: 8, L: 2, H: 1, M: 0 }, 10, 3], ["nores", { V: 23, T: 5, d: 8, L: 2, H: 2, M: 12 }, 9, 2], ["nonorm", { V: 23, T: 5, d: 8, L: 2, H: 2, M: 12 }, 9, 2]]) {
  const C = L.variant(vname), r = L.mulberry32(3), m = C.makeModel(cfg, r);
  // move every weight away from its starting value, so that no gradient is zero by symmetry (the distance scores,
  // biases, gains and shifts start at 0 or 1)
  for (const key of C.KEYS) for (let i = 0; i < m[key].length; i++) m[key][i] += 0.3 * (r() - 0.5);
  const tokens = Uint16Array.from({ length: n }, () => Math.floor(r() * cfg.V)), next = Array.from({ length: n - from }, () => Math.floor(r() * cfg.V));
  const g = C.zeros(m); C.backward(m, tokens, next, C.forward(m, tokens, from), g);
  const loss = () => { const { p } = C.forward(m, tokens, from); let s = 0; next.forEach((x, k) => (s -= Math.log(p[k * cfg.V + x]))); return s; };
  const out = [];
  for (const key of C.KEYS) {
    let w = 0;
    for (let i = 0; i < m[key].length; i++) {
      const v = m[key][i], h = 1e-5; m[key][i] = v + h; const a = loss(); m[key][i] = v - h; const b = loss(); m[key][i] = v;
      const nm = (a - b) / (2 * h), an = g[key][i], den = Math.abs(nm) + Math.abs(an);
      if (den > 1e-5) w = Math.max(w, Math.abs(nm - an) / den); // smaller gradients are below what differences can resolve
    }
    out.push(`${key} ${w.toExponential(1)}`); worst = Math.max(worst, w);
  }
  console.log(vname, JSON.stringify(cfg), `n=${n} from=${from}:`, out.join("  "));
}
console.log("largest relative difference", worst.toExponential(2));
