// Pick the subset: training clips from neither official list, test clips from the official testing list,
// one clip per speaker, in a fixed order (sorted by file name). Report loudness so we can choose QUIET.
const A = require("./audioharness.js"), fs = require("fs"), path = require("path");
const DL = path.join(__dirname, "dl"), DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const list = (f) => new Set(fs.readFileSync(path.join(DL, f), "utf8").split("\n").filter(Boolean));
const testing = list("testing_list.txt"), validation = list("validation_list.txt");
const PER_TRAIN = Number(process.env.PER_TRAIN || 400), PER_TEST = 100;
const pick = { train: [], test: [] }, stats = [];
DIGITS.forEach((w, label) => {
  const files = fs.readdirSync(path.join(DL, w)).filter((f) => f.endsWith(".wav")).sort();
  for (const [set, want, ok] of [["train", PER_TRAIN, (n) => !testing.has(n) && !validation.has(n)], ["test", PER_TEST, (n) => testing.has(n)]]) {
    const seen = new Set(); let got = 0;
    for (const f of files) {
      const name = `${w}/${f}`, speaker = f.split("_")[0];
      if (!ok(name) || seen.has(speaker)) continue;
      const s = A.readWav(path.join(DL, name)); let peak = 0; for (const v of s) peak = Math.max(peak, Math.abs(v));
      stats.push(peak);
      seen.add(speaker); pick[set].push({ name, label, speaker, peak }); got++;
      if (got >= want) break;
    }
    if (got < want) console.log("only", got, set, "speakers for", w);
  }
});
const sp = new Set(pick.train.map((p) => p.speaker)), overlap = pick.test.filter((p) => sp.has(p.speaker)).length;
stats.sort((a, b) => a - b);
console.log("train", pick.train.length, "test", pick.test.length, "| test clips whose speaker is also in training:", overlap);
console.log("peak amplitude percentiles 1/5/50%:", [0.01, 0.05, 0.5].map((q) => stats[Math.floor(q * stats.length)].toFixed(3)).join(" "));
fs.writeFileSync(path.join(__dirname, "data", "pick.json"), JSON.stringify(pick));
