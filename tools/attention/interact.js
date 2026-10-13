// Click through the page's controls in headless Chrome and print what they show: Tool 2's measurement without distance
// scores, the question with other names, Tool 1's chips and heads, the secrets' presets, and the playground (training
// for 40 seconds, then writing). Prints console messages at the end. node interact.js
const { open, sleep } = require("../words/cdp.js");
(async () => {
  const P = await open(process.env.URL || "http://localhost:8765/attention/", { width: 1200, height: 1000 });
  const ev = (x) => P.evaluate(x), txt = (id) => ev(`document.getElementById(${JSON.stringify(id)}).innerText`);
  const click = (sel) => ev(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return ${JSON.stringify("missing " + sel)}; e.click(); return "ok"; })()`);
  for (let i = 0; i < 150; i++) { await sleep(1000); if (await ev(`!/Measuring|Counting|Loading/.test(["ladder-after","sw-read","iv-read","ko-read"].map((id) => document.getElementById(id).textContent).join(" "))`)) break; }
  console.log("t2-off", await click("#t2-off"));
  for (let i = 0; i < 60; i++) { await sleep(1000); if (!/Measuring/.test(await txt("t2-off-read")) && (await txt("t2-off-read")).length > 5) break; }
  console.log("t2-off-read:", await txt("t2-off-read"));
  for (const v of ["fred", "zarko", "tom"]) { await click(`#qt-pre button[data-v="${v}"]`); await sleep(300); console.log(`qt ${v}:`, await txt("qt-read")); }
  await click('#t1-toks [data-i="13"]'); await click('#t1-head button[data-v="5"]'); await sleep(200); console.log("t1:", await txt("t1-read"));
  await click('#sw-pre button[data-v="fred"]'); await click("#sw-next"); await sleep(400); console.log("sw:", await txt("sw-read"));
  await click('#iv-pre button[data-v="quilla"]'); await click("#iv-next"); await sleep(400); console.log("iv:", await txt("iv-read"));
  await click('#ko-head button[data-v="2"]'); await sleep(200); console.log("ko:", await txt("ko-read"));
  await click('#hd-when button[data-v="3"]'); await sleep(200); console.log("hd:", await txt("hd-read"));
  await click('#pl-head button[data-v="5"]'); await click('#pl-toks [data-i="10"]'); await sleep(300); console.log("pl-wt:", await txt("pl-wt-read"));
  await click("#pl-write"); await sleep(1500); console.log("written:", await txt("pl-written"));
  if (process.env.TRAIN === "0") { console.log("console:\n" + P.logs.join("\n")); P.close(); return; }
  await click("#pl-train"); const t0 = Date.now(); await sleep(40000);
  console.log(`after ${((Date.now() - t0) / 1000).toFixed(0)} s:`, await txt("pl-status"));
  await click("#pl-train"); await sleep(1500); console.log("paused:", await txt("pl-status"));
  await click("#pl-write"); await sleep(1500); console.log("own written:", await txt("pl-written"));
  console.log("console:\n" + P.logs.join("\n"));
  P.close();
})();
