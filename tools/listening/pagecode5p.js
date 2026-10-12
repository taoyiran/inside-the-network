// Chapter 5′'s own code, pulled out of the page, run in node with the page's data and weights files.
const fs = require("fs"), zlib = require("zlib"), path = require("path"), R = path.join(__dirname, "..", "..", "listening") + "/";
const html = fs.readFileSync(R + "index.html", "utf8");
const cnn = html.match(/<script id="cnn">([\s\S]*?)<\/script>/)[1], audio = html.match(/<script id="audio">([\s\S]*?)<\/script>/)[1];
const main = html.match(/<script>\n\(function \(\) \{([\s\S]*?)<\/script>/)[1];
const grab = (a, b) => { const i = main.indexOf(a), j = main.indexOf(b, i); if (i < 0 || j < 0) throw new Error("grab " + a); return main.slice(i, j); };
const win = {}; new Function("window", fs.readFileSync(R + "sounds.js", "utf8") + fs.readFileSync(R + "weights.js", "utf8"))(win);
const atob = (s) => Buffer.from(s, "base64").toString("binary");
const code = cnn + "\n" + audio + "\n" + grab("  const bytes = (b64)", "  async function gunzip") + grab("  const HAVE_FILES", "  async function loadData") + grab("  const clipFor", "  // ---------- sound in, sound out");
const ctx = {};
new Function("ctx", "ITN_SOUNDS", "ITN_WEIGHTS", "atob", code + "\nObject.assign(ctx,{grid,forward,backward,softmax,trainStep,spectrogram,DATA,unpack,specOf,MU,NET,PLAIN,SHIFTED,toInput,makeCNN,makePlain,trainer,probsOf,argmax,moved,mulberry32,gaussian});")(ctx, win.ITN_SOUNDS, win.ITN_WEIGHTS, atob);
ctx.DATA.train = ctx.unpack(zlib.gunzipSync(Buffer.from(win.ITN_SOUNDS.train, "base64")));
ctx.DATA.test = ctx.unpack(zlib.gunzipSync(Buffer.from(win.ITN_SOUNDS.test, "base64")));
ctx.DATA.ready = true;
module.exports = ctx;
