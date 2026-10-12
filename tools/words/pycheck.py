# Run by pycheck.js: compares the page's Python translation (tok.py, nwm.py in the given folder) with the JavaScript results.
import json, sys, os, time
d = sys.argv[1]
sys.path.insert(0, d)
import tok, nwm
F = json.load(open(os.path.join(d, "fix.json")))
print("tidy mismatches:", sum(tok.tidy(s) != F["tidy"][i] for i, s in enumerate(F["raw"])))
T = tok.make_tokenizer([tuple(m) for m in F["merges"]])
print("encode mismatches:", sum(T["encode"](s) != F["ids"][i] for i, s in enumerate(F["raw"])), "| decode mismatches:", sum(T["decode"](F["ids"][i]) != F["decoded"][i] for i in range(len(F["raw"]))))
print("learned merges identical:", [list(x) for x in tok.learn_merges(dict(F["counts"]), 300)] == F["small"])
m = {"V": 2000, "k": 8, "d": 32, "h": 128, **{k: list(v) for k, v in F["init"].items()}}
adam = nwm.make_adam(m)
losses = [nwm.train_step(m, [(c, n) for c, n in b], 0.01, adam) for b in F["batches"]]
print("losses:", " ".join(f"{l:.6f}" for l in losses), "| JavaScript:", " ".join(f"{l:.6f}" for l in F["losses"]))
worst = max(abs(a - b) / (abs(b) + 1e-3) for k in nwm.KEYS for a, b in zip(m[k], F["after"][k]))
print(f"largest relative weight difference after 3 Adam steps: {worst:.1e}")
_, _, p = nwm.forward(m, F["batches"][0][0][0])
print(f"largest probability difference: {max(abs(a - b) for a, b in zip(p[:50], F['p0'])):.1e}")
