// Screenshots of every scene of the two step-by-step animations (each paused about 3/4 of the way through), and of the
// toy with a dragged query. node shots_anim.js [w=390] [dark]   → work/shots/anim/
const { open, sleep } = require("../words/cdp.js"), path = require("path"), fs = require("fs"), { html } = require("./lib.js");
// each scene's duration, read from the page's scene lists (awScenes, tfScenes)
const scenes = (name) => [...html.match(new RegExp(`const ${name} = \\[(.*?)\\];`))[1].matchAll(/d: (\d+)/g)].map((x) => +x[1]);
(async () => {
  const dark = process.argv.includes("dark"), width = Number((process.argv.find((a) => a.startsWith("w=")) || "w=1200").slice(2));
  const out = path.join(__dirname, "work", "shots", "anim"); fs.mkdirSync(out, { recursive: true });
  const P = await open(process.env.URL || "http://localhost:8765/attention/", { width, height: 1000, dark });
  await sleep(4000);
  const durs = { aw: scenes("awScenes"), tf: scenes("tfScenes") }, only = (process.argv.find((a) => a.startsWith("only=")) || "only=aw,tf").slice(5).split(",");
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
  // drag the query toward "dog", then toward the short key of "loves" (which a longer key beats)
  await P.evaluate(`document.getElementById("aw-cv").scrollIntoView({ block: "center", behavior: "instant" })`);
  await sleep(800);
  await P.evaluate(`document.querySelector("#aw-chips [data-i='5']").click()`); await sleep(400); // a step that shows the query
  const L = JSON.parse(await P.evaluate(`document.getElementById("aw-cv").dataset.compass`)); // the page's own layout
  for (const [name, qx, qy] of [["dog", 1.6, 0.3], ["loves", 0.9, 0.8]]) {
    const pt = await P.evaluate(`(() => { const r = document.getElementById("aw-cv").getBoundingClientRect(); return [r.left, r.top]; })()`); // the caption above may have moved it
    const x = pt[0] + L.cx + qx * L.unit, y = pt[1] + L.cy - qy * L.unit;
    for (const type of ["mousePressed", "mouseMoved", "mouseReleased"]) await P.send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1 });
    await sleep(300);
    await P.shot(path.join(out, `aw_drag_${name}${dark ? "_dark" : ""}${width !== 1200 ? "_" + width : ""}.png`), "#aw");
    console.log(`drag toward ${name}:`, await P.evaluate(`document.getElementById("aw-cap").innerText`));
  }
  console.log("console:\n" + P.logs.join("\n"));
  P.close();
})();
