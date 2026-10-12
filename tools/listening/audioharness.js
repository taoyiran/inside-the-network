// Loads the page's audio code (and chapter 5's network code) into node, plus a WAV reader.
const fs = require("fs"), path = require("path");
const CNN = fs.readFileSync(path.join(__dirname, "..", "seeing", "cnn.js"), "utf8");
const ctx = {};
new Function("ctx", CNN + "\n" + fs.readFileSync(path.join(__dirname, "audio.js"), "utf8") + "\nObject.assign(ctx,{grid,forward,backward,softmax,trainStep,fft,MEL,HANN,spectrogram,RATE,WIN,HOP,FRAMES,BANDS});")(ctx);
// 16-bit PCM mono WAV → Float64Array in [−1, 1)
ctx.readWav = (file) => {
  const b = fs.readFileSync(file); let o = 12, fmt = null, data = null;
  while (o < b.length) { const id = b.toString("ascii", o, o + 4), n = b.readUInt32LE(o + 4); if (id === "fmt ") fmt = { ch: b.readUInt16LE(o + 10), rate: b.readUInt32LE(o + 12), bits: b.readUInt16LE(o + 22) }; if (id === "data") data = b.subarray(o + 8, o + 8 + n); o += 8 + n + (n & 1); }
  if (!fmt || fmt.ch !== 1 || fmt.rate !== 16000 || fmt.bits !== 16) throw new Error("unexpected format " + JSON.stringify(fmt) + " " + file);
  const s = new Float64Array(data.length / 2); for (let i = 0; i < s.length; i++) s[i] = data.readInt16LE(i * 2) / 32768;
  return s;
};
module.exports = ctx;
