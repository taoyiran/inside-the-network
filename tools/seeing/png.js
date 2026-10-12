// Minimal grayscale/RGB PNG writer for looking at results.
const zlib = require("zlib"), fs = require("fs");
const crcTable = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc = (buf) => { let c = 0xffffffff; for (const b of buf) c = crcTable[(c ^ b) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function chunk(type, data) { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); }
// rgb: (x, y) => [r, g, b]
function writePNG(file, W, H, rgb) {
  const raw = Buffer.alloc((W * 3 + 1) * H);
  for (let y = 0; y < H; y++) { raw[y * (W * 3 + 1)] = 0; for (let x = 0; x < W; x++) { const [r, g, b] = rgb(x, y); const o = y * (W * 3 + 1) + 1 + x * 3; raw[o] = r; raw[o + 1] = g; raw[o + 2] = b; } }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  fs.writeFileSync(file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]));
}
// diverging color: blue positive, orange negative, white 0 (t in [-1, 1])
const div = (t) => { t = Math.max(-1, Math.min(1, t)); const pos = [42, 120, 214], neg = [235, 104, 52], a = Math.abs(t); const c = t >= 0 ? pos : neg; return c.map((v) => Math.round(255 + (v - 255) * a)); };
const gray = (v) => { const g = Math.round(255 * (1 - Math.max(0, Math.min(1, v)))); return [g, g, g]; };
// tiles: list of { data, w, h, kind: "div"|"gray", scale }, laid out in a row with gaps, each tile upscaled by `zoom`
function tilesPNG(file, rows, zoom = 6, gap = 4) {
  const rowH = rows.map((r) => Math.max(...r.map((t) => t.h)) * zoom), rowW = rows.map((r) => r.reduce((s, t) => s + t.w * zoom + gap, gap));
  const W = Math.max(...rowW), H = rowH.reduce((s, h) => s + h + gap, gap);
  const px = new Array(W * H).fill(null).map(() => [255, 255, 255]);
  let oy = gap;
  rows.forEach((r, ri) => { let ox = gap; for (const t of r) { const m = t.kind === "div" ? Math.max(1e-9, ...Array.from(t.data, Math.abs)) : 1; for (let y = 0; y < t.h * zoom; y++) for (let x = 0; x < t.w * zoom; x++) { const v = t.data[Math.floor(y / zoom) * t.w + Math.floor(x / zoom)]; px[(oy + y) * W + ox + x] = t.kind === "div" ? div(v / m) : gray(v); } ox += t.w * zoom + gap; } oy += rowH[ri] + gap; });
  writePNG(file, W, H, (x, y) => px[y * W + x]);
}
module.exports = { writePNG, tilesPNG, div, gray };
