// The loudest band level (dB) of each clip, to choose the QUIET floor below every real clip.
const A = require("./audioharness.js"), fs = require("fs"), path = require("path");
const pick = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "pick.json"), "utf8"));
const vals = [];
for (const p of pick.train.concat(pick.test).slice(0, 5000)) {
  const s = A.readWav(path.join(__dirname, "dl", p.name)), re = new Float64Array(512), im = new Float64Array(512); let top = -Infinity;
  for (let t = 0; t < 32; t++) { for (let n = 0; n < 512; n++) { const i = t * 500 + n; re[n] = (i < s.length ? s[i] : 0) * A.HANN[n]; im[n] = 0; } A.fft(re, im); for (const w of A.MEL) { let pw = 0; w.forEach((v, k) => { if (v) pw += v * (re[k] * re[k] + im[k] * im[k]); }); top = Math.max(top, 10 * Math.log10(pw + 1e-10)); } }
  vals.push(top);
}
vals.sort((a, b) => a - b);
console.log("loudest band level (dB): min", vals[0].toFixed(1), "1%", vals[Math.floor(0.01 * vals.length)].toFixed(1), "median", vals[vals.length >> 1].toFixed(1), "max", vals.at(-1).toFixed(1));
// silence and a quiet hiss for comparison
const hiss = Float64Array.from({ length: 16000 }, () => (Math.random() - 0.5) * 0.002);
let top = -Infinity; { const re = new Float64Array(512), im = new Float64Array(512); for (let t = 0; t < 32; t++) { for (let n = 0; n < 512; n++) { re[n] = hiss[t * 500 + n] * A.HANN[n] || 0; im[n] = 0; } A.fft(re, im); for (const w of A.MEL) { let pw = 0; w.forEach((v, k) => { if (v) pw += v * (re[k] * re[k] + im[k] * im[k]); }); top = Math.max(top, 10 * Math.log10(pw + 1e-10)); } } }
console.log("a quiet hiss (±0.001) reaches", top.toFixed(1), "dB");
