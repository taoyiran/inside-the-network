// Write words/stories.js and words/weights.js.
// node package.js Q128 600000   (the run's checkpoints in work/models/, its log in work/runs/<run>.txt)
const L = require("./lib.js"), fs = require("fs"), path = require("path"), zlib = require("zlib");
const run = process.argv[2] || "Q128", finalStep = +process.argv[3] || 600000, out = process.env.OUT || path.join(__dirname, "..", "..", "words");
const merges = L.merges(), train = L.loadTokens("train.bin"), testAll = L.loadTokens("test.bin");
let cut = 0, n = 0; while (n < 1000) { if (testAll[cut] === 0) n++; cut++; } // the first 1,000 test stories
const test = testAll.slice(0, cut);
const b64gz = (u) => zlib.gzipSync(Buffer.from(u.buffer, u.byteOffset, u.byteLength), { level: 9 }).toString("base64");
fs.writeFileSync(path.join(out, "stories.js"), `// 21,000 stories from the TinyStories dataset by Ronen Eldan and Yuanzhi Li (Microsoft Research),
// https://huggingface.co/datasets/roneneldan/TinyStories, paper: R. Eldan and Y. Li, "TinyStories: How Small Can
// Language Models Be and Still Speak Coherent English?", arXiv:2305.07759 (2023).
// Made available under the Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0):
// https://cdla.dev/sharing-1-0/ . This file is shared under the same agreement.
// THIS FILE HAS BEEN CHANGED from the original data: it holds stories 2 to 20,001 of TinyStoriesV2-GPT4-train.txt
// (train) and the first 1,000 complete stories of TinyStoriesV2-GPT4-valid.txt (test; the file opens with the end of a
// story, which is skipped), each with surrounding blank space removed, lowercased,
// tidied (curly quotes, dashes; other characters removed) and cut into tokens by this chapter's own code.
// merges: the 1,922 merges learned from the training stories, in order ("a ##b" pairs separated by "|").
// train, test: the token numbers, each story followed by token 0 (the end of a story), as 16-bit little-endian
// numbers, then gzip, then base64.
window.ITN_STORIES = {
  merges: ${JSON.stringify(merges.map((m) => m.join(" ")).join("|"))},
  train: "${b64gz(train)}",
  test: "${b64gz(test)}",
};
`);
const f32 = (arrs) => { const total = arrs.reduce((a, x) => a + x.length, 0), b = Buffer.alloc(total * 4); let o = 0; for (const a of arrs) for (const v of a) { b.writeFloatLE(v, o); o += 4; } return b.toString("base64"); };
const m = L.loadModel(`${run}_${finalStep}.bin`), tk = L.tokenizer();
const MAP = "cat dog bird fish bunny frog bear red blue green yellow pink purple brown mom dad brother sister baby grandma friend happy sad scared angry excited tired proud park house garden forest school beach kitchen ran jumped played walked laughed smiled cried".split(" ");
const ids = MAP.map((w) => tk.id.get(w)), rowsOf = (E) => ids.flatMap((t) => Array.from(E.subarray(t * m.d, (t + 1) * m.d)));
// the map words' rows at moments of the same run; step 0 is the starting table (seed 1)
const init = L.NWM.makeModel({ V: m.V, k: m.k, d: m.d, h: m.h }, L.mulberry32(1));
const steps = [0, 1000, 3000, 10000, 40000, 150000, finalStep].filter((s, i, a) => s <= finalStep && a.indexOf(s) === i);
const moments = steps.map((s) => [s, s * 64, f32([rowsOf(s ? L.loadModel(`${run}_${s}.bin`).E : init.E)])]);
const log = fs.readFileSync(L.work("runs", run + ".txt"), "utf8").split("\n").map((l) => l.match(/step (\d+) .*?(?:test|validation loss) ([\d.]+) .*?(?:acc|accuracy) ([\d.]+)%/)).filter(Boolean).map((x) => [+x[1], +x[2], +x[3]]).filter((x) => x[0] <= finalStep);
fs.writeFileSync(path.join(out, "weights.js"), `// The network trained in advance with this chapter's own code, on the 20,000 TinyStories training stories:
// 8 tokens in, a table of 2,000 rows of 32, 128 tanh neurons, 2,000 scores. Its weights, rounded to 32-bit numbers
// (little-endian, then base64): E, W1, b1, W2, b2, in that order.
// moments: [step, examples seen, the rows of the 42 map words (mapWords) at that step], from the same training run.
// log: [step, loss, accuracy %] measured during that run on 20,000 tokens of the validation stories 1,001 to 2,000
// (other stories than the 1,000 test stories the page uses).
window.ITN_WEIGHTS = {
  shape: { V: ${m.V}, k: ${m.k}, d: ${m.d}, h: ${m.h} },
  steps: ${finalStep},
  net: "${f32([m.E, m.W1, m.b1, m.W2, m.b2])}",
  mapWords: ${JSON.stringify(MAP.join(" "))},
  moments: ${JSON.stringify(moments)},
  log: ${JSON.stringify(log)},
};
`);
console.log("stories.js", fs.statSync(path.join(out, "stories.js")).size, "bytes | weights.js", fs.statSync(path.join(out, "weights.js")).size, "bytes | moments", steps.join(", "));
