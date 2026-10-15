// Minimal CDP driver: launch headless Chrome, open the page, wait, run expressions, take screenshots.
const { spawn } = require("child_process"), fs = require("fs"), path = require("path");
const PORT = Number(process.env.PORT || 9641), DIR = path.join(__dirname, "work", "chrome-profile-" + PORT);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function open(url, { width = 1200, height = 900, dark = false, inject = "" } = {}) {
  const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${DIR}`, `--window-size=${width},${height}`, "--no-first-run", "--autoplay-policy=no-user-gesture-required", "--mute-audio", "about:blank"], { stdio: "ignore" });
  let ws;
  for (let i = 0; i < 50; i++) { try { const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); const pg = list.find((t) => t.type === "page"); if (pg) { ws = new WebSocket(pg.webSocketDebuggerUrl); break; } } catch (e) {} await sleep(200); }
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let id = 0; const pend = new Map(), logs = [];
  ws.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } else if (m.method === "Runtime.consoleAPICalled") logs.push(m.params.type + ": " + m.params.args.map((a) => a.value ?? a.description).join(" ")); else if (m.method === "Runtime.exceptionThrown") logs.push("EXC: " + JSON.stringify(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text)); });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  await send("Runtime.enable"); await send("Page.enable");
  await send("Network.enable"); await send("Network.setCacheDisabled", { cacheDisabled: true }); // the profile persists: always load the files as they are now
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: dark ? "dark" : "light" }] });
  if (inject) await send("Page.addScriptToEvaluateOnNewDocument", { source: inject });
  await send("Page.navigate", { url });
  const evaluate = async (expr) => { const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.result.exceptionDetails) return "ERR " + JSON.stringify(r.result.exceptionDetails.exception?.description); return r.result.result.value; };
  const shot = async (file, selector) => {
    let clip;
    if (selector) { const b = await evaluate(`(() => { const e = document.querySelector(${JSON.stringify(selector)}); e.scrollIntoView({block: "center"}); const r = e.getBoundingClientRect(); return [r.x + scrollX, r.y + scrollY, r.width, r.height]; })()`); await sleep(400); clip = { x: b[0], y: b[1], width: b[2], height: b[3], scale: 1 }; }
    const r = await send("Page.captureScreenshot", clip ? { format: "png", clip, captureBeyondViewport: true } : { format: "png" });
    fs.writeFileSync(file, Buffer.from(r.result.data, "base64"));
  };
  return { send, evaluate, shot, logs, close: () => { try { ws.close(); } catch (e) {} chrome.kill(); } };
}
module.exports = { open, sleep };
