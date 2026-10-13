// Switch heads off, one at a time (its value table set to 0, so it brings back 16 zeros), and measure right first
// guesses on every token of the page's 100 test stories, by where the answer last appeared (as compare.js):
// near (among the last 8 tokens), far (earlier in the story, further back), name (the story's name, far), new;
// and how many places get a name as the first guess, and how many of those are wrong.
// node ablate.js F8_19000.bin [heads=3,8]   (heads=…: switch these off together instead; heads=R: set every distance score to 0)
const L = require("./lib.js");
const K = 8, NS = 100;
const NAMES = L.names();
const m0 = L.loadModel(process.argv[2]), { stories } = L.storiesOf("test.bin"), test = stories.slice(0, NS);
// categories per position
const cats = test.map(({ tokens, next }) => {
  const X = L.nameOf(tokens);
  return Array.from(next, (ans, r) => {
    let w = "new"; for (let i = r; i >= 1; i--) if (tokens[i] === ans) { w = r + 1 - i <= K ? "near" : "far"; break; }
    return { w, name: ans === X && w === "far" };
  });
});
function measure(m) {
  const right = { all: 0, near: 0, far: 0, name: 0, new: 0 }, n = { all: 0, near: 0, far: 0, name: 0, new: 0 }; let guessed = 0, wrong = 0;
  test.forEach(({ tokens, next }, s) => {
    const f = L.ATN.forward(m, tokens);
    for (let r = 0; r < next.length; r++) {
      let b = 0; for (let t = 1; t < m.V; t++) if (f.p[r * m.V + t] > f.p[r * m.V + b]) b = t;
      const ok = b === next[r], c = cats[s][r];
      if (NAMES.includes(b)) { guessed++; if (!ok) wrong++; }
      for (const k of ["all", c.w].concat(c.name ? ["name"] : [])) { n[k]++; right[k] += ok; }
    }
  });
  return { ...Object.fromEntries(Object.keys(n).map((k) => [k, right[k] / n[k]])), names: [guessed, wrong] };
}
const off = (heads) => { const m = { ...m0, Wv: Float64Array.from(m0.Wv) }; for (const h of heads) m.Wv.fill(0, h * m.dv * m.d, (h + 1) * m.dv * m.d); return m; };
const fmt = (r) => Object.entries(r).map(([k, v]) => (k === "names" ? `| a name guessed at ${v[0]} places, ${v[1]} wrongly` : `${k} ${(100 * v).toFixed(1)}%`)).join("  ");
console.log("all heads on:     ", fmt(measure(m0)));
const arg = process.argv.find((a) => a.startsWith("heads="));
if (arg === "heads=R") { const m = { ...m0, R: new Float64Array(m0.R.length) }; console.log("distance scores 0:".padEnd(18), fmt(measure(m))); }
else if (arg) { const hs = arg.slice(6).split(",").map((x) => +x - 1); console.log(`heads ${hs.map((h) => h + 1).join(",")} off:`.padEnd(18), fmt(measure(off(hs)))); }
else for (let h = 0; h < m0.H; h++) console.log(`head ${h + 1} off:`.padEnd(18), fmt(measure(off([h]))));
