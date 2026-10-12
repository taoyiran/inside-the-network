import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import train_step, grid
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcheck_in.json")))
net = d["net"]
def img(px):
    g = grid(1, 28, 28); g["d"] = list(px); return g
batch = [(img(px), y) for px, y in d["batch"]]
out = [train_step(net, batch[3:] if s % 2 else batch[:3], 0.1) for s in range(3)]
js = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xcheck_js.json")))
print("PY:", " ".join(f"{o[0]:.12f}" for o in out))
print("max loss diff", max(abs(a[0] - b["loss"]) for a, b in zip(out, js["out"])), "| accuracy equal", all(a[1] == b["accuracy"] for a, b in zip(out, js["out"])))
wd = max(abs(a - b) for L, J in zip(net, js["net"]) if J for key in ("W", "b") for a, b in zip(L[key], J[key]))
print("max weight diff after 3 steps", wd)
