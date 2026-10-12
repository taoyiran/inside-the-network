// A small convolutional network, written out by hand. An image is a stack of grids of numbers:
// c channels, each h × w, stored in one flat array. Channel ch, row y, column x is at (ch·h + y)·w + x.
const grid = (c, h, w) => ({ c, h, w, d: new Float64Array(c * h * w) });

// A convolution layer: each filter is a small k × k stencil of weights (one per input channel).
// It slides over every position of the input; at each one it multiplies and adds, giving one output number.
// The same weights are used at every position, so a filter finds its pattern wherever it appears.
function convForward(L, input) {
  const { c, h, w } = input, k = L.k, oh = h - k + 1, ow = w - k + 1, out = grid(L.n, oh, ow);
  for (let f = 0; f < L.n; f++) for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
    let sum = L.b[f];
    for (let ch = 0; ch < c; ch++) for (let i = 0; i < k; i++) for (let j = 0; j < k; j++)
      sum += L.W[((f * c + ch) * k + i) * k + j] * input.d[(ch * h + y + i) * w + x + j];
    out.d[(f * oh + y) * ow + x] = sum;
  }
  return out;
}

// ReLU keeps positive numbers and turns negative ones into 0. Max pooling keeps the largest number
// in each 2 × 2 block, so the grid halves in size and a pattern can shift a pixel without changing much.
function reluForward(input) {
  const out = grid(input.c, input.h, input.w);
  input.d.forEach((v, i) => { out.d[i] = v > 0 ? v : 0; });
  return out;
}
function poolForward(input) {
  const { c, h, w } = input, out = grid(c, h / 2, w / 2);
  out.from = new Int32Array(out.d.length); // where each maximum came from, for the backward pass
  for (let ch = 0; ch < c; ch++) for (let y = 0; y < h / 2; y++) for (let x = 0; x < w / 2; x++) {
    let best = -Infinity, at = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      const p = (ch * h + 2 * y + i) * w + 2 * x + j;
      if (input.d[p] > best) { best = input.d[p]; at = p; }
    }
    const o = (ch * (h / 2) + y) * (w / 2) + x;
    out.d[o] = best; out.from[o] = at;
  }
  return out;
}

// The last layer is an ordinary dense layer: every number in the final stack votes for each of the
// 10 digits. Softmax turns the 10 scores into probabilities that add up to 1.
function denseForward(L, input) {
  const out = grid(L.n, 1, 1), m = input.d.length;
  for (let j = 0; j < L.n; j++) { let sum = L.b[j]; for (let i = 0; i < m; i++) sum += L.W[j * m + i] * input.d[i]; out.d[j] = sum; }
  return out;
}
function softmax(scores) {
  const top = Math.max(...scores), e = Array.from(scores, (s) => Math.exp(s - top)), total = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / total);
}
const LAYER = { conv: convForward, relu: (L, x) => reluForward(x), pool: (L, x) => poolForward(x), dense: denseForward };
// The forward pass keeps every layer's output: the backward pass needs them.
function forward(net, image) {
  const values = [image];
  for (const L of net) values.push(LAYER[L.type](L, values[values.length - 1]));
  return values;
}

// The backward pass for one image walks back through the layers with the chain rule. Each layer gets the
// gradient of the loss with respect to its output (g), adds up the gradients of its own weights, and hands
// back the gradient with respect to its input. A filter's weight gradient sums over every position it visited.
function backward(net, values, label, grads) {
  const p = softmax(values[values.length - 1].d);
  let g = Float64Array.from(p, (v, j) => v - (j === label ? 1 : 0)); // softmax with cross-entropy: p − y
  for (let l = net.length - 1; l >= 0; l--) {
    const L = net[l], input = values[l], out = values[l + 1], gIn = new Float64Array(input.d.length);
    if (L.type === "dense") {
      const m = input.d.length;
      for (let j = 0; j < L.n; j++) { grads[l].b[j] += g[j]; for (let i = 0; i < m; i++) { grads[l].W[j * m + i] += g[j] * input.d[i]; gIn[i] += L.W[j * m + i] * g[j]; } }
    } else if (L.type === "relu") {
      for (let i = 0; i < gIn.length; i++) gIn[i] = input.d[i] > 0 ? g[i] : 0;
    } else if (L.type === "pool") {
      for (let o = 0; o < g.length; o++) gIn[out.from[o]] += g[o];
    } else {
      const { c, h, w } = input, k = L.k, oh = out.h, ow = out.w;
      for (let f = 0; f < L.n; f++) for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
        const go = g[(f * oh + y) * ow + x];
        if (go === 0) continue;
        grads[l].b[f] += go;
        for (let ch = 0; ch < c; ch++) for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
          const wi = ((f * c + ch) * k + i) * k + j, xi = (ch * h + y + i) * w + x + j;
          grads[l].W[wi] += go * input.d[xi];
          gIn[xi] += L.W[wi] * go;
        }
      }
    }
    g = gIn;
  }
  return g; // the gradient with respect to the image's pixels
}

// One step of training on a small batch of images: add up their gradients, average, and step against them.
// With thousands of images, a step on a random batch of 20 is far cheaper than a step on all of them.
function trainStep(net, batch, lr) {
  const grads = net.map((L) => (L.W ? { W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) } : null));
  let loss = 0, right = 0;
  for (const { image, label } of batch) {
    const values = forward(net, image), p = softmax(values[values.length - 1].d);
    loss -= Math.log(Math.max(p[label], 1e-300));
    if (p.indexOf(Math.max(...p)) === label) right++;
    backward(net, values, label, grads);
  }
  net.forEach((L, l) => { if (!L.W) return; const n = batch.length; L.W.forEach((v, i) => { L.W[i] -= (lr * grads[l].W[i]) / n; }); L.b.forEach((v, i) => { L.b[i] -= (lr * grads[l].b[i]) / n; }); });
  return { loss: loss / batch.length, accuracy: right / batch.length };
}
