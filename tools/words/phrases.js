// For phrases read from standard input: what counting answers (and from how many tokens), and a network's top 5.
// printf '%s\n' "the giant and the little fox" | node phrases.js Q128_600000.bin
const L = require("./lib.js"), fs = require("fs");
const tk = L.tokenizer(), m = L.loadModel(process.argv[2]), tok = L.loadTokens("train.bin"), K = 8;
for (const ph of fs.readFileSync(0, "utf8").split("\n").map((s) => s.trim()).filter(Boolean)) {
  const ids = tk.encode(ph), c = new Array(K).fill(0); for (let j = 1; j <= Math.min(K, ids.length); j++) c[K - j] = ids[ids.length - j];
  const cr = c.slice().reverse(), roll = new Array(K).fill(0), counts = Array.from({ length: K + 1 }, () => new Map()), times = new Array(K + 1).fill(0);
  for (let t = 0; t < tok.length; t++) { for (let l = 1; l <= K; l++) { if (roll[l - 1] !== cr[l - 1]) break; times[l]++; counts[l].set(tok[t], (counts[l].get(tok[t]) || 0) + 1); } if (tok[t] === 0) roll.fill(0); else { for (let j = K - 1; j > 0; j--) roll[j] = roll[j - 1]; roll[0] = tok[t]; } }
  const n = Math.min(K, ids.length); let used = 0; for (let l = n; l >= 1; l--) if (times[l]) { used = l; break; }
  const top3 = (mm) => [...mm].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([x, v]) => `${tk.vocab[x]} (${v})`).join(", ");
  const p = L.NWM.forward(m, c).p, top = [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 5).map((x) => `${tk.vocab[x]} ${(100 * p[x]).toFixed(0)}%`).join(", ");
  console.log(`${ph}\n   counting: the last ${used} of ${n} tokens, seen ${times[used]} times → ${top3(counts[used])}\n   network: ${top}`);
}
