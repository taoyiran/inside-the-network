// Open the chapter in headless Chrome (needs Google Chrome and the folder served, e.g. python3 -m http.server 8765
// from the repo), wait for the live measurements, print every readout and console message, and save screenshots
// to work/shots/. node look.js [dark] [w=390] [shots=story,ladder,...]   (URL=… to check another address)
const { open, sleep } = require("../words/cdp.js"), path = require("path"), fs = require("fs");
(async () => {
  const dark = process.argv.includes("dark"), width = Number((process.argv.find((a) => a.startsWith("w=")) || "w=1200").slice(2));
  const url = process.env.URL || "http://localhost:8765/attention/", out = path.join(__dirname, "work", "shots"); fs.mkdirSync(out, { recursive: true });
  const P = await open(url, { width, height: 1000, dark });
  const done = `!/Measuring|Counting|Loading/.test(["ladder-after", "sw-read", "iv-read", "hd-read", "ko-read"].map((id) => document.getElementById(id).textContent).join(" "))`;
  for (let i = 0; i < 120; i++) { await sleep(1000); if (await P.evaluate(done)) break; }
  await sleep(1000);
  for (const id of ["st-read", "ladder", "ladder-split", "ladder-after", "f-cap", "qt-read", "t1-read", "t2-read", "t3-read", "hd-read", "hd-after", "sw-read", "sw-after", "ko-read", "ko-after", "iv-read", "iv-after", "gc-check", "runs-train", "pl-status", "pl-wt-read"]) console.log(id.padEnd(12), String(await P.evaluate(`(document.getElementById(${JSON.stringify(id)}) || {}).innerText`)).replace(/\n/g, " | "));
  const sel = { story: "#story", ladder: "#ladder", film: "#film", question: "#question", tool1: "#tool1", tool2: "#tool2", tool3: "#tool3", heads: "#heads", swap: "#swap", knock: "#knock", invent: "#invent", playground: "#playground", algo: "#algo" };
  const shots = (process.argv.find((a) => a.startsWith("shots=")) || "shots=" + Object.keys(sel).join(",")).slice(6).split(",").filter(Boolean);
  for (const s of shots) await P.shot(path.join(out, `${s}${dark ? "_dark" : ""}${width !== 1200 ? "_" + width : ""}.png`), sel[s]);
  console.log("console:\n" + P.logs.join("\n"));
  P.close();
})();
