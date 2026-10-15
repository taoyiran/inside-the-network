// Secret 2: each of the GPT's heads switched off in turn (its value table set to 0, so it adds nothing to the rows),
// measured as the page measures it: invented names (one name per name story; never-seen pairs and all pieces, at later
// mentions), the second copy of repeated tokens, and every token of the page's 100 test stories.
// node ablate.js G_9500.bin [pairs=a,b]   (pairs: also switch off two heads at once, given as block·4 + head numbers)
const L = require("./lib.js");
const m = L.loadModel(process.argv[2]), { d, L: NL, H } = m, dk = d / H, test = L.storiesOf("test.bin").stories.slice(0, 100);
const off = (heads) => { const n = { ...m, Wv: Float64Array.from(m.Wv) }; for (const k of heads) { const l = Math.floor(k / H), h = k % H; for (let c = 0; c < dk; c++) for (let e = 0; e < d; e++) n.Wv[l * d * d + (h * dk + c) * d + e] = 0; } return n; };
const name = (k) => `block ${Math.floor(k / H) + 1} head ${(k % H) + 1}`, pc = (x) => (100 * x).toFixed(1).padStart(5) + "%";
const row = (label, net) => { const r = L.pageMeasures(net), e = L.evaluate(net, test); console.log(`${label.padEnd(22)} names never-seen later ${pc(r.unseen.later)} (all ${pc(r.all.later)}) | second copy ${pc(r.second)} | all test tokens ${pc(e.acc)}`); };
row("none off", m);
const pairs = (process.argv.find((a) => a.startsWith("pairs=")) || "pairs=").slice(6).split(",").filter(Boolean).map(Number);
if (pairs.length) row(`${pairs.map(name).join(" + ")} off`, off(pairs));
for (let k = 0; k < NL * H; k++) row(`${name(k)} off`, off([k]));
