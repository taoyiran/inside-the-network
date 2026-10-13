# Run by pycheck.js: compares the page's Python translation (attn.py in the given folder) with the JavaScript results.
import json, sys, os
d = sys.argv[1]
sys.path.insert(0, d)
import attn
F = json.load(open(os.path.join(d, "fix.json")))
cfg = F["cfg"]
m = {**cfg, **{k: list(v) for k, v in F["init"].items()}}
f = attn.forward(m, F["batches"][0][0][0], 5)
p = f["p"][0][:40]
print(f"forward from position 5: largest probability difference {max(abs(a - b) for a, b in zip(p, F['p0'])):.1e}")
H, T = cfg["H"], cfg["T"]
w = [f["w"][r][hd][t] if t < len(f["w"][r][hd]) else 0.0 for r in range(3) for hd in range(H) for t in range(T)]
print(f"attention weights: largest difference {max(abs(a - b) for a, b in zip(w, F['w0'])):.1e}")
adam = attn.make_adam(m)
losses = [attn.train_step(m, [(t, n) for t, n in b], 0.01, adam) for b in F["batches"]]
print("losses:", " ".join(f"{l:.9f}" for l in losses), "| JavaScript:", " ".join(f"{l:.9f}" for l in F["losses"]))
worst = max(abs(a - b) / (abs(b) + 1e-3) for k in attn.KEYS for a, b in zip(m[k], F["after"][k]))
print(f"largest relative weight difference after 3 Adam steps: {worst:.1e}")
