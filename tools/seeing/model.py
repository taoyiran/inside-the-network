# A small convolutional network, written out by hand. An image is a stack of grids of numbers:
# c channels, each h × w, stored in one flat list. Channel ch, row y, column x is at (ch·h + y)·w + x.
import math
def grid(c, h, w):
    return {"c": c, "h": h, "w": w, "d": [0.0] * (c * h * w)}

# A convolution layer: each filter is a small k × k stencil of weights (one per input channel).
# It slides over every position of the input; at each one it multiplies and adds, giving one output number.
# The same weights are used at every position, so a filter finds its pattern wherever it appears.
def conv_forward(L, inp):
    c, h, w, k = inp["c"], inp["h"], inp["w"], L["k"]
    oh, ow = h - k + 1, w - k + 1
    out = grid(L["n"], oh, ow)
    for f in range(L["n"]):
        for y in range(oh):
            for x in range(ow):
                total = L["b"][f]
                for ch in range(c):
                    for i in range(k):
                        for j in range(k):
                            total += L["W"][((f * c + ch) * k + i) * k + j] * inp["d"][(ch * h + y + i) * w + x + j]
                out["d"][(f * oh + y) * ow + x] = total
    return out

# ReLU keeps positive numbers and turns negative ones into 0. Max pooling keeps the largest number
# in each 2 × 2 block, so the grid halves in size and a pattern can shift a pixel without changing much.
def relu_forward(inp):
    out = grid(inp["c"], inp["h"], inp["w"])
    out["d"] = [v if v > 0 else 0.0 for v in inp["d"]]
    return out
def pool_forward(inp):
    c, h, w = inp["c"], inp["h"], inp["w"]
    out = grid(c, h // 2, w // 2)
    out["from"] = [0] * len(out["d"])  # where each maximum came from, for the backward pass
    for ch in range(c):
        for y in range(h // 2):
            for x in range(w // 2):
                best, at = -math.inf, 0
                for i in range(2):
                    for j in range(2):
                        p = (ch * h + 2 * y + i) * w + 2 * x + j
                        if inp["d"][p] > best:
                            best, at = inp["d"][p], p
                o = (ch * (h // 2) + y) * (w // 2) + x
                out["d"][o], out["from"][o] = best, at
    return out

# The last layer is an ordinary dense layer: every number in the final stack votes for each of the
# 10 digits. Softmax turns the 10 scores into probabilities that add up to 1.
def dense_forward(L, inp):
    out, m = grid(L["n"], 1, 1), len(inp["d"])
    for j in range(L["n"]):
        out["d"][j] = L["b"][j] + sum(L["W"][j * m + i] * inp["d"][i] for i in range(m))
    return out
def softmax(scores):
    top = max(scores)
    e = [math.exp(s - top) for s in scores]
    total = sum(e)
    return [v / total for v in e]
LAYER = {"conv": conv_forward, "relu": lambda L, x: relu_forward(x), "pool": lambda L, x: pool_forward(x), "dense": dense_forward}
# The forward pass keeps every layer's output: the backward pass needs them.
def forward(net, image):
    values = [image]
    for L in net:
        values.append(LAYER[L["type"]](L, values[-1]))
    return values

# The backward pass for one image walks back through the layers with the chain rule. Each layer gets the
# gradient of the loss with respect to its output (g), adds up the gradients of its own weights, and hands
# back the gradient with respect to its input. A filter's weight gradient sums over every position it visited.
def backward(net, values, label, grads):
    p = softmax(values[-1]["d"])
    g = [v - (1 if j == label else 0) for j, v in enumerate(p)]  # softmax with cross-entropy: p − y
    for l in range(len(net) - 1, -1, -1):
        L, inp, out = net[l], values[l], values[l + 1]
        g_in = [0.0] * len(inp["d"])
        if L["type"] == "dense":
            m = len(inp["d"])
            for j in range(L["n"]):
                grads[l]["b"][j] += g[j]
                for i in range(m):
                    grads[l]["W"][j * m + i] += g[j] * inp["d"][i]
                    g_in[i] += L["W"][j * m + i] * g[j]
        elif L["type"] == "relu":
            g_in = [gv if v > 0 else 0.0 for v, gv in zip(inp["d"], g)]
        elif L["type"] == "pool":
            for o, gv in enumerate(g):
                g_in[out["from"][o]] += gv
        else:
            c, h, w, k, oh, ow = inp["c"], inp["h"], inp["w"], L["k"], out["h"], out["w"]
            for f in range(L["n"]):
                for y in range(oh):
                    for x in range(ow):
                        go = g[(f * oh + y) * ow + x]
                        if go == 0:
                            continue
                        grads[l]["b"][f] += go
                        for ch in range(c):
                            for i in range(k):
                                for j in range(k):
                                    wi, xi = ((f * c + ch) * k + i) * k + j, (ch * h + y + i) * w + x + j
                                    grads[l]["W"][wi] += go * inp["d"][xi]
                                    g_in[xi] += L["W"][wi] * go
        g = g_in
    return g  # the gradient with respect to the image's pixels

# One step of training on a small batch of images: add up their gradients, average, and step against them.
# With thousands of images, a step on a random batch of 20 is far cheaper than a step on all of them.
def train_step(net, batch, lr):
    grads = [{"W": [0.0] * len(L["W"]), "b": [0.0] * len(L["b"])} if "W" in L else None for L in net]
    loss, right = 0.0, 0
    for image, label in batch:
        values = forward(net, image)
        p = softmax(values[-1]["d"])
        loss -= math.log(max(p[label], 1e-300))
        right += p.index(max(p)) == label
        backward(net, values, label, grads)
    for L, gr in zip(net, grads):
        if gr is None:
            continue
        L["W"] = [v - lr * gv / len(batch) for v, gv in zip(L["W"], gr["W"])]
        L["b"] = [v - lr * gv / len(batch) for v, gv in zip(L["b"], gr["b"])]
    return loss / len(batch), right / len(batch)
