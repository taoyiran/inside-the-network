# Run by pycheck.js: compares the page's Python translation (gpt.py in the given folder) with the JavaScript results.
import json, sys, os
d = sys.argv[1]
sys.path.insert(0, d)
import gpt
F = json.load(open(os.path.join(d, "fix.json")))
cfg = F["cfg"]
L, d_ = cfg["L"], cfg["d"]
# the JavaScript keeps each kind of weight in one array, block after block; the Python keeps a list per block
def unflat(arrs): return {k: [v[l * len(v) // L:(l + 1) * len(v) // L] for l in range(L)] if k in gpt.PER_BLOCK else list(v) for k, v in arrs.items()}
def flat(m): return {k: [x for part in m[k] for x in part] if k in gpt.PER_BLOCK else m[k] for k in gpt.KEYS}
seeded = [0]
def mulberry32(a):
    def r():
        a_ = r.a = (r.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a_ ^ (a_ >> 15)) * (1 | a_)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    r.a = a
    return r
fresh = flat(gpt.make_model(**cfg, rand=mulberry32(7)))
print(f"make_model against makeModel, same seed: largest difference {max(abs(a - b) for k in gpt.KEYS for a, b in zip(fresh[k], F['fresh'][k])):.1e}")
m = {**cfg, **unflat(F["init"])}
f = gpt.forward(m, F["batches"][0][0][0], 5)
print(f"forward from position 5: largest probability difference {max(abs(a - b) for a, b in zip(f['p'][:40], F['p0'])):.1e}")
H, T = cfg["H"], cfg["T"]
w = [f["blocks"][2]["w"][i][hd][t] if t < len(f["blocks"][2]["w"][i][hd]) else 0.0 for i in range(15) for hd in range(H) for t in range(T)]
print(f"block 3's attention weights: largest difference {max(abs(a - b) for a, b in zip(w, F['w0'])):.1e}")
print(f"rows out of the last block: largest difference {max(abs(a - b) for a, b in zip(f['xs'][L][:15 * d_], F['x0'])):.1e}")
adam = gpt.make_adam(m)
losses = [gpt.train_step(m, [(t, n) for t, n in b], 0.01, adam) for b in F["batches"]]
print("losses:", " ".join(f"{l:.9f}" for l in losses), "| JavaScript:", " ".join(f"{l:.9f}" for l in F["losses"]))
after = flat(m)
worst = max(abs(a - b) / (abs(b) + 1e-3) for k in gpt.KEYS for a, b in zip(after[k], F["after"][k]))
print(f"largest relative weight difference after 3 Adam steps: {worst:.1e}")
