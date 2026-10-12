// Learn the 1,922 merges from the training stories and cut the stories into tokens.
// Training: stories 2 to 20,001 of the training file (the loader skips the text before the first <|endoftext|>).
// Test: the first 2,000 complete stories of the validation file; the page carries the first 1,000, and
// stories 1,001 to 2,000 are used only to choose among trained networks.
// Writes work/merges.json, work/train.bin and work/test.bin (16-bit tokens, each story followed by 0).
const L = require("./lib.js"), fs = require("fs");
const train = L.stories(L.work("dl", "train_head.txt"), 20000), test = L.stories(L.work("dl", "valid.txt"), 2000);
const counts = new Map();
for (const s of train) for (const p of L.TOK.pieces(s)) if (/^[a-z]/.test(p)) counts.set(p, (counts.get(p) || 0) + 1);
const merges = L.TOK.learnMerges(counts, 2000 - 78), tk = L.TOK.makeTokenizer(merges);
const flat = (set) => { const out = []; for (const s of set) out.push(...tk.encode(s), 0); return Uint16Array.from(out); };
const tr = flat(train), te = flat(test);
fs.writeFileSync(L.work("merges.json"), JSON.stringify(merges));
fs.writeFileSync(L.work("train.bin"), Buffer.from(tr.buffer));
fs.writeFileSync(L.work("test.bin"), Buffer.from(te.buffer));
let whole = 0, words = 0; for (const s of test.slice(0, 1000)) for (const p of L.TOK.pieces(s)) if (/^[a-z]/.test(p)) { words++; if (tk.encode(p).length === 1) whole++; }
console.log(`${counts.size} different words in training; ${merges.length} merges; vocabulary ${tk.size}; ${tr.length} training tokens; ${te.length} test tokens; ${(100 * whole / words).toFixed(2)}% of the words in the first 1,000 test stories are one token`);
