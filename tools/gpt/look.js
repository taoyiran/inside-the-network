// Open the chapter in headless Chrome (needs Google Chrome and the folder served, e.g. python3 -m http.server 8765
// from the repo), wait for the live measurements, print every readout and console message, and save screenshots
// to work/shots/. node look.js [dark] [w=390] [shots=case,ladder,...] [wait=180]   (URL=… to check another address)
const { open, sleep } = require("../words/cdp.js"), path = require("path"), fs = require("fs");
(async () => {
  const dark = process.argv.includes("dark"), width = Number((process.argv.find((a) => a.startsWith("w=")) || "w=1200").slice(2)), wait = Number((process.argv.find((a) => a.startsWith("wait=")) || "wait=240").slice(5));
  const url = process.env.URL || "http://localhost:8765/gpt/", out = path.join(__dirname, "work", "shots"); fs.mkdirSync(out, { recursive: true });
  const P = await open(url, { width, height: 1000, dark });
  const done = `!/Measuring|Counting|Loading/.test(["ladder-after", "cs-read", "iv-read", "hd-read", "ko-after", "ln-read"].map((id) => document.getElementById(id).textContent).join(" "))`;
  const t0 = Date.now();
  for (let i = 0; i < wait; i++) { await sleep(1000); if (await P.evaluate(done)) break; }
  console.log(`measurements done after ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  await sleep(1000);
  for (const id of ["cs-read", "ladder", "ladder-split", "ladder-after", "tw-cap", "film-cap", "qt-read", "t1-read", "t2-read", "t3-read", "iv-read", "iv-agg", "iv-after", "hd-read", "hd-after", "ko-read", "ko-after", "wn-read", "wn-after", "lv-read", "ln-table", "ln-read", "ln-after", "rs-read", "rs-after", "gc-check", "runs-train", "pl-status", "pl-wt-read"]) console.log(id.padEnd(12), String(await P.evaluate(`(document.getElementById(${JSON.stringify(id)}) || {}).innerText`)).replace(/\n/g, " | "));
  const sel = { case: "#case", ladder: "#ladder", tw: "#tw", film: "#film", question: "#question", tool1: "#tool1", tool2: "#tool2", tool3: "#tool3", tf: "#tf", invent: "#invent", heads: "#heads", knock: "#knock", when: "#when", live: "#live", lens: "#lens", resid: "#resid", playground: "#playground", algo: "#algo" };
  const shots = (process.argv.find((a) => a.startsWith("shots=")) || "shots=" + Object.keys(sel).join(",")).slice(6).split(",").filter(Boolean);
  for (const s of shots) await P.shot(path.join(out, `${s}${dark ? "_dark" : ""}${width !== 1200 ? "_" + width : ""}.png`), sel[s]);
  console.log("console:\n" + P.logs.join("\n"));
  P.close();
})();
