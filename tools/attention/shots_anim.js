// Screenshots of every scene of the two step-by-step animations (each paused about 3/4 of the way through), and of the
// toy with a dragged query. node shots_anim.js [w=390] [dark]   → work/shots/anim/
const { open, sleep } = require("../words/cdp.js"), path = require("path"), fs = require("fs");
(async () => {
  const dark = process.argv.includes("dark"), width = Number((process.argv.find((a) => a.startsWith("w=")) || "w=1200").slice(2));
  const out = path.join(__dirname, "work", "shots", "anim"); fs.mkdirSync(out, { recursive: true });
  const P = await open(process.env.URL || "http://localhost:8765/attention/", { width, height: 1000, dark });
  await sleep(4000);
  const durs = { aw: [6500, 8500, 6500, 13000, 9500, 11000, 8000, 12000], tf: [9000, 10000, 11000, 15000, 11000, 12000, 14000] }, only = (process.argv.find((a) => a.startsWith("only=")) || "only=aw,tf").slice(5).split(",");
  for (const id of only) {
    for (let i = 0; i < durs[id].length; i++) {
      await P.evaluate(`document.querySelector("#${id}-chips [data-i='${i}']").click()`);
      await sleep(durs[id][i] * 0.75);
      await P.evaluate(`(() => { const b = document.getElementById("${id}-play"); if (b.textContent === "Pause") b.click(); })()`);
      await sleep(150);
      await P.shot(path.join(out, `${id}${i}${dark ? "_dark" : ""}${width !== 1200 ? "_" + width : ""}.png`), "#" + id);
      console.log(id, i, (await P.evaluate(`document.getElementById("${id}-cap").innerText`)).slice(0, 90));
    }
  }
  // drag the query toward "dog"
  await P.evaluate(`document.getElementById("aw-cv").scrollIntoView({ block: "center", behavior: "instant" })`);
  await sleep(800);
  const pt = await P.evaluate(`(() => { const r = document.getElementById("aw-cv").getBoundingClientRect(); return [r.left, r.top, r.width]; })()`);
  await P.evaluate(`document.querySelector("#aw-chips [data-i='5']").click()`); await sleep(400); // a step that shows the query
  const L = JSON.parse(await P.evaluate(`document.getElementById("aw-cv").dataset.compass`)); // the page's own layout
  const x = pt[0] + L.cx + 1.6 * L.unit, y = pt[1] + L.cy - 0.3 * L.unit;
  for (const type of ["mousePressed", "mouseMoved", "mouseReleased"]) await P.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1 });
  await sleep(300);
  await P.shot(path.join(out, `aw_drag${dark ? "_dark" : ""}${width !== 1200 ? "_" + width : ""}.png`), "#aw");
  console.log("drag:", (await P.evaluate(`document.getElementById("aw-cap").innerText`)).slice(0, 200));
  console.log("console:\n" + P.logs.join("\n"));
  P.close();
})();
