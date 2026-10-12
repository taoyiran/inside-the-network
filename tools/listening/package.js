// listening/sounds.js and listening/weights.js
const A = require("./audioharness.js"), fs = require("fs"), path = require("path"), zlib = require("zlib");
const R = process.argv[2], pick = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "pick.json"), "utf8"));
const pack = (set) => { const b = fs.readFileSync(path.join(__dirname, "data", set + ".bin")), n = b.length / 1025, out = Buffer.alloc(n * 512 + n);
  for (let i = 0; i < n * 512; i++) out[i] = (b[2 * i] << 4) | b[2 * i + 1];
  b.copy(out, n * 512, n * 1024); return zlib.gzipSync(out, { level: 9 }).toString("base64"); };
// 8-bit μ-law: y = sign(x)·ln(1 + 255|x|)/ln 256, stored as round(127·y) + 128
const mulaw = (s) => { const o = Buffer.alloc(16000); for (let i = 0; i < 16000; i++) { const x = Math.max(-1, Math.min(1, s[i] || 0)), y = (Math.sign(x) * Math.log1p(255 * Math.abs(x))) / Math.log(256); o[i] = Math.round(127 * y) + 128; } return o; };
const clips = [];
for (let d = 0; d < 10; d++) {
  const tr = pick.train.map((p, k) => ({ p, k })).filter((o) => o.p.label === d), step = Math.floor(tr.length / 24);
  for (const i of [0, 1, 2]) clips.push({ set: "train", k: tr[i * step].k, label: d, name: tr[i * step].p.name });
  const te = pick.test.map((p, k) => ({ p, k })).filter((o) => o.p.label === d);
  for (const i of [0, 1]) clips.push({ set: "test", k: te[i].k, label: d, name: te[i].p.name });
}
const three = pick.test.map((p, k) => ({ p, k })).find((o) => o.p.label === 3 && o.p.peak > 0.3);
if (!clips.some((c) => c.set === "test" && c.k === three.k)) clips.push({ set: "test", k: three.k, label: 3, name: three.p.name });
// test recordings, which the networks listen to, as exact 16-bit samples (so their spectrograms match the stored ones);
// training recordings, which are only played, as 8-bit μ-law
const pcm16 = (s) => { const o = Buffer.alloc(32000); for (let i = 0; i < 16000; i++) o.writeInt16LE(Math.round((s[i] || 0) * 32768), i * 2); return o; };
const audio = Buffer.concat(clips.map((c) => (c.set === "test" ? pcm16 : mulaw)(A.readWav(path.join(__dirname, "dl", c.name)))));
const js = `// 9,000 spoken digits from the Speech Commands dataset (v0.02) by Pete Warden, Google, made available under the
// Creative Commons Attribution 4.0 license (https://creativecommons.org/licenses/by/4.0/); source:
// https://www.tensorflow.org/datasets/catalog/speech_commands; paper: P. Warden, "Speech Commands: A Dataset for
// Limited-Vocabulary Speech Recognition", arXiv:1804.03209 (2018). This file holds a subset, changed as follows:
// train: 800 clips of each digit (8,000), test: 100 of each (1,000) from the official testing list, one clip per speaker.
// Each is stored as its 32 × 32 spectrogram (16 levels, two per byte), made by this chapter's own code; then the labels.
// clips: ${clips.length} of the recordings themselves, 1 second at 16 kHz: test recordings as 16-bit samples, training
// recordings (for listening only) in 8-bit μ-law; clipIndex says which, in order.
// Each string is gzip, then base64.
window.ITN_SOUNDS = {
  train: "${pack("train")}",
  test: "${pack("test")}",
  clips: "${zlib.gzipSync(audio, { level: 9 }).toString("base64")}",
  clipIndex: ${JSON.stringify(clips.map((c) => [c.set, c.k, c.label]))},
};
`;
fs.writeFileSync(path.join(R, "listening", "sounds.js"), js);
const b64net = (f) => { const net = JSON.parse(fs.readFileSync(path.join(__dirname, f), "utf8")), vals = net.flatMap((L) => (L.W ? L.W.concat(L.b) : [])), buf = Buffer.alloc(vals.length * 8); vals.forEach((v, i) => buf.writeDoubleLE(v, i * 8)); return [buf.toString("base64"), vals.length]; };
const [c, nc] = b64net("final_cnn.json"), [m, nm] = b64net("final_plain.json"), [g] = b64net("final_aug.json");
fs.writeFileSync(path.join(R, "listening", "weights.js"), `// The two networks trained in advance with this chapter's own code, as exact 64-bit numbers (little-endian, then base64).
// They were trained on Speech Commands recordings (CC BY 4.0, Pete Warden, Google).
// cnn: the convolutional network (${nc} weights and biases), plain: the plain network (${nm}), shifted: the same
// convolutional network trained on pictures moved up or down by up to 3 bands; layer by layer, weights then biases.
window.ITN_WEIGHTS = {
  cnn: "${c}",
  plain: "${m}",
  shifted: "${g}",
};
`);
console.log("clips", clips.length, "| sounds.js", fs.statSync(path.join(R, "listening", "sounds.js")).size, "bytes | weights.js", fs.statSync(path.join(R, "listening", "weights.js")).size, "bytes | params", nc, nm);
