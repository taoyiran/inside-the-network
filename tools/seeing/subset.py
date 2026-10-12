# Read MNIST idx files and write a balanced subset as raw bytes: images (N×784 uint8) followed by labels (N uint8).
import gzip, struct, sys, os
dl, out = sys.argv[1], sys.argv[2]
def images(p):
    with gzip.open(p) as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16)); assert magic == 2051 and r == c == 28
        return n, f.read(n * 784)
def labels(p):
    with gzip.open(p) as f:
        magic, n = struct.unpack(">II", f.read(8)); assert magic == 2049
        return f.read(n)
def subset(img, lab, per):
    n, data = img; count = [0] * 10; pick = []
    for i in range(n):
        y = lab[i]
        if count[y] < per: count[y] += 1; pick.append(i)
    return bytes(b for i in pick for b in data[i*784:(i+1)*784]), bytes(lab[i] for i in pick), pick
for name, ip, lp, per in [("train", "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 600), ("test", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 100)]:
    x, y, pick = subset(images(os.path.join(dl, ip)), labels(os.path.join(dl, lp)), per)
    open(os.path.join(out, name + ".bin"), "wb").write(x + y)
    print(name, len(y), "images; source indices", pick[0], "…", pick[-1])
