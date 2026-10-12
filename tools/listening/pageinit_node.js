// Runs the page's own code (chapter 5 model + 5′ audio + 5′ training init) in node, with the data in page format.
const fs = require("fs"), path = require("path");
const CNN = fs.readFileSync(path.join(__dirname, "..", "seeing", "cnn.js"), "utf8"), AUDIO = fs.readFileSync(path.join(__dirname, "audio.js"), "utf8"), INIT = fs.readFileSync(path.join(__dirname, "init.js"), "utf8");
const DATA = { train: null, test: null };
const load = (set) => { const b = fs.readFileSync(path.join(__dirname, "data", set + ".bin")), n = b.length / 1025; return { q: b.subarray(0, n * 1024), labels: b.subarray(n * 1024), n }; };
DATA.train = load("train"); DATA.test = load("test");
const ctx = {};
new Function("ctx", "DATA", CNN + "\n" + AUDIO + "\nconst specOf = (set, k) => { const g = grid(1, BANDS, FRAMES); for (let i = 0; i < 1024; i++) g.d[i] = set.q[k * 1024 + i] / 15; return g; };\n" + INIT + "\nObject.assign(ctx,{grid,forward,backward,softmax,trainStep,spectrogram,fft,MEL,HANN,toInput,makeCNN,makePlain,trainer,specOf,moved,mulberry32,gaussian,RATE,WIN,HOP,FRAMES,BANDS,QUIET});")(ctx, DATA);
ctx.DATA = DATA;
ctx.accuracy = (net, set = DATA.test) => { let ok = 0; for (let k = 0; k < set.n; k++) { const s = ctx.forward(net, ctx.toInput(ctx.specOf(set, k))).at(-1).d; let b = 0; for (let j = 1; j < 10; j++) if (s[j] > s[b]) b = j; if (b === set.labels[k]) ok++; } return ok / set.n; };
module.exports = ctx;
