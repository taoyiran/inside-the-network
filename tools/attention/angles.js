// Attention's score is not chapter 6's closeness (the cosine of two words' rows). Over every pair "token i reads
// token j" (j ≤ i, fewer than 256 back) in the page's 100 test stories, for each head: how the dot-product part of
// the score, q_i · k_j / √8, relates to the cosine of the two tokens' rows; how well score(a reads b) predicts
// score(b reads a); and, inside the head's own query/key space, how often the key closest in angle to the query is
// not the one with the highest dot product (the lengths count).
// node angles.js (the shipped attention/weights.js, as the page has it), or node angles.js F8_19000.bin (a checkpoint)
const L = require("./lib.js");
const m = process.argv[2] ? L.loadModel(process.argv[2]) : L.loadShipped(), tk = L.tokenizer();
const { d, H, dk, T, E, Wq, Wk } = m, sc = 1 / Math.sqrt(dk);
const vs = L.storiesOf("test.bin").stories.slice(0, 100);
// per token: its row's unit vector, and per head its query and key and their lengths
const unit = new Float64Array(m.V * d), Q = new Float64Array(m.V * H * dk), K = new Float64Array(m.V * H * dk), QL = new Float64Array(m.V * H), KL = new Float64Array(m.V * H);
for (let t = 0; t < m.V; t++) {
  let n = 0; for (let e = 0; e < d; e++) n += E[t * d + e] ** 2; n = Math.sqrt(n);
  for (let e = 0; e < d; e++) unit[t * d + e] = E[t * d + e] / n;
  for (let c = 0; c < H * dk; c++) { let a = 0, b = 0; for (let e = 0; e < d; e++) { a += Wq[c * d + e] * E[t * d + e]; b += Wk[c * d + e] * E[t * d + e]; } Q[t * H * dk + c] = a; K[t * H * dk + c] = b; }
  for (let h = 0; h < H; h++) { let a = 0, b = 0; for (let c = 0; c < dk; c++) { a += Q[(t * H + h) * dk + c] ** 2; b += K[(t * H + h) * dk + c] ** 2; } QL[t * H + h] = Math.sqrt(a); KL[t * H + h] = Math.sqrt(b); }
}
const cosRow = (a, b) => { let s = 0; for (let e = 0; e < d; e++) s += unit[a * d + e] * unit[b * d + e]; return s; };
const qk = (h, a, b) => { let s = 0; for (let c = 0; c < dk; c++) s += Q[(a * H + h) * dk + c] * K[(b * H + h) * dk + c]; return s * sc; };
// running Pearson correlation
const corr = () => ({ n: 0, x: 0, y: 0, xx: 0, yy: 0, xy: 0, add(a, b) { this.n++; this.x += a; this.y += b; this.xx += a * a; this.yy += b * b; this.xy += a * b; }, get r() { const n = this.n; return (this.xy - (this.x * this.y) / n) / Math.sqrt((this.xx - this.x ** 2 / n) * (this.yy - this.y ** 2 / n)); } });
const ranks = (a) => { const o = a.map((v, i) => i).sort((x, y) => a[x] - a[y]), r = new Float64Array(a.length); o.forEach((i, k) => (r[i] = k)); return r; };
const spearman = (x, y) => { const n = x.length, a = ranks(x), b = ranks(y), mid = (n - 1) / 2; let s = 0, u = 0, v = 0; for (let i = 0; i < n; i++) { s += (a[i] - mid) * (b[i] - mid); u += (a[i] - mid) ** 2; v += (b[i] - mid) ** 2; } return u && v ? s / Math.sqrt(u * v) : 1; };
const rowCorr = Array.from({ length: H }, corr), pairs = new Set();
const same = Array.from({ length: H }, () => ({ n: 0, above: 0 })), angleMiss = new Float64Array(H), rankAgree = new Float64Array(H);
let N = 0, P = 0, ranked = 0;
for (const { tokens } of vs) for (let i = 0; i < tokens.length; i++) {
  N++;
  const a = tokens[i], n = Math.min(i, T - 1) + 1, cs = new Float64Array(n); // the cosine of the rows: the same for every head
  for (let t = 0; t < n; t++) { cs[t] = cosRow(a, tokens[i - t]); P++; pairs.add(a * m.V + tokens[i - t]); }
  if (n >= 10) ranked++;
  for (let h = 0; h < H; h++) {
    const dots = new Float64Array(n), coss = new Float64Array(n); let best = 0, near = 0;
    for (let t = 0; t < n; t++) {
      const b = tokens[i - t], s = qk(h, a, b);
      rowCorr[h].add(s, cs[t]); dots[t] = s; coss[t] = s / sc / QL[a * H + h] / KL[b * H + h];
      if (s > dots[best]) best = t;
      if (coss[t] > coss[near]) near = t;
    }
    // an earlier copy of the same token (cosine of rows 1): is its score above the median of the scores this position gives?
    const med = Float64Array.from(dots).sort()[Math.floor(n / 2)];
    for (let t = 1; t < n; t++) if (tokens[i - t] === a) { same[h].n++; if (dots[t] > med) same[h].above++; }
    if (tokens[i - best] !== tokens[i - near]) angleMiss[h]++;
    if (n >= 10) rankAgree[h] += spearman(Array.from(dots), Array.from(coss)); // the angle's order of the keys against the dot product's
  }
}
// asymmetry over the distinct ordered pairs of token types that meet in the stories
const asym = Array.from({ length: H }, corr);
for (const p of pairs) { const a = Math.floor(p / m.V), b = p % m.V; if (a === b) continue; for (let h = 0; h < H; h++) asym[h].add(qk(h, a, b), qk(h, b, a)); }
// how much the keys' lengths vary, over all 2,000 tokens
for (let h = 0; h < H; h++) { const ls = []; for (let t = 0; t < m.V; t++) ls.push(KL[t * H + h]); ls.sort((x, y) => x - y); const p = (f) => ls[Math.floor(f * (ls.length - 1))]; console.log(`head ${h + 1}: key length 10th percentile ${p(0.1).toFixed(2)}, median ${p(0.5).toFixed(2)}, 90th ${p(0.9).toFixed(2)} (${(p(0.9) / p(0.1)).toFixed(1)}-fold)`); }
console.log(`${N} reading positions, ${P} pairs, ${pairs.size} distinct ordered token pairs`);
console.log(`head | corr(q·k/√8, cosine of rows) | corr(a reads b, b reads a) | an earlier copy of the same token: above the median | top key by dot product is not the closest in angle | rank correlation of dot product and angle's cosine, per position (mean over the ${ranked} positions with 10 or more keys)`);
for (let h = 0; h < H; h++) console.log(`  ${h + 1}  |  ${rowCorr[h].r.toFixed(2)}  |  ${asym[h].r.toFixed(2)}  |  ${((100 * same[h].above) / same[h].n).toFixed(0)}% of ${same[h].n}  |  ${((100 * angleMiss[h]) / N).toFixed(1)}%  |  ${(rankAgree[h] / ranked).toFixed(2)}`);
// the pairs the page names under Tool 1
for (const [h, a, b] of [[6, "to", "lily"], [6, "to", "red"], [6, "to", "to"], [6, "tom", "lily"], [6, "tom", "ball"], [6, "to", "tom"], [6, "tom", "to"]]) {
  const x = tk.id.get(a), y = tk.id.get(b); if (x == null || y == null) { console.log(`no token ${a} or ${b}`); continue; }
  console.log(`head ${h}: “${a}” reads “${b}”: q·k/√8 ${qk(h - 1, x, y).toFixed(2)}, cosine of rows ${cosRow(x, y).toFixed(2)}`);
}
