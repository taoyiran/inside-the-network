// Does a pronoun's rebuilt input carry the story's name? In the test stories with a one-token name, at every position
// that reads "she", "he", "her" or "his" and whose last mention of the name is more than 8 tokens back:
//   1. per head, the average weight on the name's mentions (all of them) against the even share they would get;
//   2. per head, from the 16 numbers the head brings back there, which name's value they are closest to (cosine),
//      among the one-token names: how often it is the story's own name.
// node pronouns.js F8_19000.bin   (NSTORIES=1000)
const L = require("./lib.js");
const NS = +(process.env.NSTORIES || 1000), K = 8;
const tk = L.tokenizer(), V = tk.vocab;
const NAMES = L.names();
const PRON = ["she", "he", "her", "his"].map((w) => tk.id.get(w));
const m = L.loadModel(process.argv[2]), { H, T, d, dv } = m;
const { stories } = L.storiesOf("test.bin");
// a token's value for head h
const valueOf = (t, h) => { const out = new Float64Array(dv); for (let c = 0; c < dv; c++) { let s = 0; for (let e = 0; e < d; e++) s += m.Wv[(h * dv + c) * d + e] * m.E[t * d + e]; out[c] = s; } return out; };
const cos = (a, b) => { let s = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return s / Math.sqrt(x * y); };
const nameVals = Array.from({ length: H }, (_, h) => NAMES.map((t) => valueOf(t, h)));
let n = 0; const onName = new Float64Array(H), even = new Float64Array(H), hit = new Float64Array(H);
stories.slice(0, NS).forEach(({ tokens }) => {
  const X = L.nameOf(tokens);
  if (X < 0) return;
  const xi = NAMES.indexOf(X);
  const f = L.ATN.forward(m, tokens);
  for (let r = 1; r < tokens.length; r++) {
    if (!PRON.includes(tokens[r])) continue;
    let last = -1; for (let i = r; i >= 1; i--) if (tokens[i] === X) { last = i; break; }
    if (last < 1 || r - last < K) continue;
    n++;
    let mentions = 0; for (let t = 0; t <= Math.min(r, T - 1); t++) if (tokens[r - t] === X) mentions++;
    for (let h = 0; h < H; h++) {
      let w = 0; for (let t = 0; t <= Math.min(r, T - 1); t++) if (tokens[r - t] === X) w += f.w[(r * H + h) * T + t];
      onName[h] += w; even[h] += mentions / (Math.min(r, T - 1) + 1);
      const o = f.u.subarray(r * (d + H * dv) + d + h * dv, r * (d + H * dv) + d + (h + 1) * dv);
      let b = 0, bs = -2; nameVals[h].forEach((v, j) => { const c = cos(o, v); if (c > bs) { bs = c; b = j; } });
      if (b === xi) hit[h]++;
    }
  }
});
console.log(`${n} pronoun positions whose story's name was last mentioned more than 8 tokens back`);
for (let h = 0; h < H; h++) console.log(`head ${h + 1}: weight on the name ${(100 * onName[h] / n).toFixed(1)}% (even share ${(100 * even[h] / n).toFixed(1)}%) | closest name value is the story's name ${(100 * hit[h] / n).toFixed(1)}% (chance ${(100 / NAMES.length).toFixed(1)}%)`);
