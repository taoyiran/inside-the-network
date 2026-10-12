// Spectrograms for the picked clips with the page's own function; uint8 output; loudest-dB statistics.
const A = require("./audioharness.js"), fs = require("fs"), path = require("path");
const pick = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "pick.json"), "utf8"));
const loud = [];
for (const set of ["train", "test"]) {
  const N = pick[set].length, out = Buffer.alloc(N * 1024 + N);
  pick[set].forEach((p, i) => {
    const s = A.readWav(path.join(__dirname, "dl", p.name));
    // loudest band level before the QUIET floor (recomputed here only for the statistic)
    const sp = A.spectrogram(s); for (let q = 0; q < 1024; q++) out[i * 1024 + q] = Math.round(sp.d[q] * 15);
    out[N * 1024 + i] = p.label;
  });
  fs.writeFileSync(path.join(__dirname, "data", `${set}.bin`), out);
  console.log(set, N, "clips →", out.length, "bytes");
}
