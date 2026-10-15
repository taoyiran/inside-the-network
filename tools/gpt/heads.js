// What each of the GPT's heads does, block by block:
//   - where its weight goes in the page's 100 test stories, by distance back (the bins of chapter 7), and the
//     tokens it reads most per appearance (read at least 5 times);
//   - on repeated random tokens (<end>, 40 tokens drawn from the 1,000 most common, the same 40 again; the same 50
//     sequences as induction.js): its average weight, in the second copy, on the token just before (a
//     previous-token head) and on the token right after the earlier copy of the token being read (an induction
//     head), and on <end>.
//   - at later mentions of the page's invented names (one name per name story, as the page measures it): its average
//     weight on the token just before and on the token right after the earlier copy of the token being read. The page
//     classes a head by these: a previous-token head, or an induction head (in block 2 or above).
// node heads.js R_9500.bin
const L = require("./lib.js");
const tk = L.tokenizer(), V = tk.vocab;
const m = L.loadModel(process.argv[2]), { T, H } = m, NB = L.NB, NBL = NB.length;
const test = L.storiesOf("test.bin").stories.slice(0, 100);
const binOf = new Uint8Array(T); NB.forEach(([a, b], k) => { for (let t = a; t <= Math.min(b, T - 1); t++) binOf[t] = k; });
const bins = new Float64Array(m.L * H * NBL), reads = new Float64Array(m.L * H * 2000), seen = new Float64Array(2000); let n = 0;
for (const { tokens, next } of test) {
  const f = L.GPT.forward(m, tokens);
  for (let r = 0; r < next.length; r++) {
    n++; seen[tokens[r]]++;
    for (let l = 0; l < m.L; l++) { const w = f.blocks[l].w; for (let h = 0; h < H; h++) for (let t = 0; t <= Math.min(r, T - 1); t++) { const x = w[(r * H + h) * T + t]; bins[(l * H + h) * NBL + binOf[t]] += x; reads[(l * H + h) * 2000 + tokens[r - t]] += x; } }
  }
}
const reps = L.pageSetup().reps; // the page's 50 repeated sequences
const prev = new Float64Array(m.L * H), ind = new Float64Array(m.L * H), end = new Float64Array(m.L * H); let nr = 0;
for (const s of reps) {
  const f = L.GPT.forward(m, s);
  for (let i = 42; i < s.length; i++) { nr++; for (let l = 0; l < m.L; l++) for (let h = 0; h < H; h++) { const w = f.blocks[l].w, b = (i * H + h) * T; prev[l * H + h] += w[b + 1]; ind[l * H + h] += w[b + 39]; end[l * H + h] += w[b + i]; } }
}
// at later mentions of the invented names
const { items } = L.pageSetup(), nprev = new Float64Array(m.L * H), nind = new Float64Array(m.L * H); let nn = 0;
for (const { toks, at } of items) {
  const f = L.GPT.forward(m, toks);
  for (const { r, first } of at) {
    if (first) continue;
    let e = -1; for (let i = r - 1; i >= 1; i--) if (toks[i] === toks[r]) { e = i; break; }
    if (e < 0 || r - e - 1 >= T) continue;
    nn++; for (let l = 0; l < m.L; l++) for (let h = 0; h < H; h++) { const w = f.blocks[l].w, b = (r * H + h) * T; nprev[l * H + h] += w[b + 1]; nind[l * H + h] += w[b + r - e - 1]; }
  }
}
const pc = (x) => (100 * x).toFixed(0).padStart(3) + "%";
console.log(`${n} test positions; bins ${NB.map(([a, b]) => (a === b ? a : `${a}-${b}`)).join(" ")} | repeated tokens, second copy: before, induction, <end> | invented names, ${nn} later-mention positions: before, induction`);
for (let l = 0; l < m.L; l++) for (let h = 0; h < H; h++) {
  const k = l * H + h, b = Array.from(bins.subarray(k * NBL, (k + 1) * NBL), (x) => pc(x / n)).join("");
  const per = []; for (let t = 0; t < 2000; t++) if (seen[t] >= 5) per.push([t, reads[k * 2000 + t] / seen[t]]); per.sort((a, c) => c[1] - a[1]);
  console.log(`block ${l + 1} head ${h + 1}: ${b} | ${pc(prev[k] / nr)} ${pc(ind[k] / nr)} ${pc(end[k] / nr)} | ${pc(nprev[k] / nn)} ${pc(nind[k] / nn)} | reads most: ${per.slice(0, 6).map(([t]) => V[t] === "\n" ? "↵" : V[t]).join(" ")}`);
}
