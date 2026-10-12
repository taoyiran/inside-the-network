# Write seeing/digits.js (base64 of the gzipped subset) and seeing/weights.js (exact float64 weights).
import base64, json, struct, sys, os
C, R = sys.argv[1], sys.argv[2]
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
with open(os.path.join(R, "seeing", "digits.js"), "w") as f:
    f.write("// 7,000 handwritten digits from the MNIST database by Yann LeCun and Corinna Cortes, made available under\n"
            "// the Creative Commons Attribution-Share Alike 3.0 license (https://creativecommons.org/licenses/by-sa/3.0/);\n"
            "// this file, a subset of it, is shared under the same license. The first 600 of each digit from MNIST's\n"
            "// training set (6,000) and the first 100 of each from its test set (1,000).\n"
            "// Each string is gzip, then base64: N images of 28 × 28 bytes (0 = paper, 255 = ink), then N labels.\n")
    f.write("window.ITN_DIGITS = {\n  train: \"%s\",\n  test: \"%s\",\n};\n" % (b64(os.path.join(C, "data", "train.bin.gz")), b64(os.path.join(C, "data", "test.bin.gz"))))
def pack(net):
    vals = [v for L in net if "W" in L for v in L["W"] + L["b"]]
    return base64.b64encode(struct.pack("<%dd" % len(vals), *vals)).decode(), len(vals)
cnn, n1 = pack(json.load(open(os.path.join(C, "ref_weights.json"))))
mlp, n2 = pack(json.load(open(os.path.join(C, "mlp8_weights.json"))))
with open(os.path.join(R, "seeing", "weights.js"), "w") as f:
    f.write("// The two networks trained in advance with this chapter's own code, as exact 64-bit numbers (little-endian, then base64):\n"
            "// cnn: the convolutional network (%d weights and biases), mlp: the plain network (%d), layer by layer, weights then biases.\n" % (n1, n2))
    f.write("window.ITN_WEIGHTS = {\n  cnn: \"%s\",\n  mlp: \"%s\",\n};\n" % (cnn, mlp))
print("params", n1, n2)
