// Screenshot every scene of the page's three animations (the toy, the film, the transformer diagrams), near the end of
// each scene, to work/shots/anim/. node shots_anim.js [dark] [w=390] [only=tw,film,tf]
const { open, sleep } = require("../words/cdp.js"), path = require("path"), fs = require("fs");
const SCENES = { tw: [6500, 7500, 11000, 7500, 15000, 8500, 9500, 8000, 12000], film: [8000, 11000, 12000, 10000, 10000, 9000], tf: [9000, 13000, 13000, 10000, 14000, 11000, 13000] };
(async () => {
  const dark = process.argv.includes("dark"), width = Number((process.argv.find((a) => a.startsWith("w=")) || "w=1200").slice(2));
  const only = (process.argv.find((a) => a.startsWith("only=")) || "only=tw,film,tf").slice(5).split(",");
  const out = path.join(__dirname, "work", "shots", "anim"); fs.mkdirSync(out, { recursive: true });
  const P = await open(process.env.URL || "http://localhost:8765/gpt/", { width, height: 1100, dark });
  for (let i = 0; i < 120; i++) { await sleep(1000); if (await P.evaluate(`!/Measuring/.test(document.getElementById("hd-read").textContent)`)) break; }
  for (const id of only) {
    for (let s = 0; s < SCENES[id].length; s++) {
      await P.evaluate(`(() => { document.getElementById("${id}").scrollIntoView({ block: "start" }); document.querySelectorAll("#${id}-chips button")[${s}].click(); })()`);
      await sleep(Math.max(1500, SCENES[id][s] - 700));
      await P.shot(path.join(out, `${id}_${s}${dark ? "_dark" : ""}${width !== 1200 ? "_" + width : ""}.png`), `#${id}`);
    }
  }
  console.log(P.logs.join("\n"));
  P.close();
})();
