// Load chapter 5's own code (model script + page functions) into node, with the page's data and weights files.
const fs = require("fs"), zlib = require("zlib"), path = require("path"), R = path.join(__dirname, "..", "..", "seeing") + "/";
const html = fs.readFileSync(R + "index.html", "utf8");
const model = html.match(/<script id="cnn">([\s\S]*?)<\/script>/)[1];
const main = html.match(/<script>\n\(function \(\) \{([\s\S]*?)<\/script>/)[1];
const grab = (a, b) => { const i = main.indexOf(a), j = main.indexOf(b, i); if (i < 0 || j < 0) throw new Error("grab " + a); return main.slice(i, j); };
const win = {}; new Function("window", fs.readFileSync(R + "digits.js", "utf8") + fs.readFileSync(R + "weights.js", "utf8"))(win);
const atob = (s) => Buffer.from(s, "base64").toString("binary");
const code = model + "\n" + grab("  const bytes = (b64)", "  async function gunzip") + grab("  const DATA = {", "  async function loadDigits") + grab("  // weights: little-endian", "  // ---------- training a fresh network") + grab("  function mulberry32", "  const stencilTiles") + grab("  // gradient descent on −log p(digit)", "  // ---------- work done once at load");
const ctx = {};
new Function("ctx", "ITN_WEIGHTS", "ITN_DIGITS", "atob", code + "\nObject.assign(ctx,{grid,forward,backward,softmax,trainStep,DATA,toImage,NET,PLAIN,probsOf,argmax,pixelGrad,shifted,mulberry32,gaussian,makeCNN,shuffled,trainer,dreamer,fool,averageOf});")(ctx, win.ITN_WEIGHTS, win.ITN_DIGITS, atob);
// fill DATA the way loadDigits does
const tr = zlib.gunzipSync(Buffer.from(win.ITN_DIGITS.train, "base64")), te = zlib.gunzipSync(Buffer.from(win.ITN_DIGITS.test, "base64"));
const n = tr.length / 785; ctx.DATA.train = { px: tr, labels: tr.subarray(n * 784), n };
const m = te.length / 785; for (let k = 0; k < m; k++) ctx.DATA.test.push({ image: ctx.toImage(te, k), label: te[m * 784 + k] });
for (let d = 0; d < 10; d++) ctx.DATA.firstOf[d] = ctx.DATA.test.findIndex((e) => e.label === d);
ctx.DATA.ready = true;
module.exports = ctx;
