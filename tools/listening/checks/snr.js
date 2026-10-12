// "dB below the word": the page scales the hiss to the RMS of the whole second (silence included). How much louder is the word itself?
const A = require("../audioharness.js"), fs = require("fs"), path = require("path");
const pick = JSON.parse(fs.readFileSync(path.join(__dirname, "../data/pick.json"), "utf8"));
const out = [], out2 = [];
for (const p of pick.test) {
  const s0 = A.readWav(path.join(__dirname, "../dl", p.name)); const s = s0; // page uses the clip as is
  let mean = 0; for (const v of s) mean += v; mean /= s.length;
  const tot = s.reduce((a, v) => a + v * v, 0) / s.length;
  // 10 ms frames, active = within 30 dB of the loudest frame (DC removed for the frame test)
  const fr = []; for (let i = 0; i + 160 <= s.length; i += 160) { let e = 0; for (let j = i; j < i + 160; j++) e += (s[j] - mean) ** 2; fr.push(e / 160); }
  const mx = Math.max(...fr), act = fr.filter((e) => e > mx / 1000);
  const actPow = act.reduce((a, b) => a + b, 0) / act.length;
  out.push(10 * Math.log10(actPow / tot)); out2.push(act.length / fr.length);
}
const q = (a, p) => { const b = [...a].sort((x, y) => x - y); return b[Math.floor(p * (b.length - 1))]; };
console.log("word (active frames) level above whole-second RMS, dB: p10", q(out, .1).toFixed(1), "median", q(out, .5).toFixed(1), "p90", q(out, .9).toFixed(1));
console.log("fraction of the second that is active: median", q(out2, .5).toFixed(2));
