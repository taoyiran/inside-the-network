// A tape-based autograd, small enough to read in one sitting.
// Every operation creates a Value and records, for each input,
// the local derivative ∂output/∂input. backward() walks the
// tape in reverse and applies the chain rule.
let TAPE = null;

class Value {
  constructor(data, inputs = null, local = null, op = "") {
    this.data = data;     // the number itself
    this.grad = 0;        // ∂loss/∂this, filled in by backward()
    this.inputs = inputs; // the Values this one was computed from
    this.local = local;   // ∂this/∂input, one entry per input
    this.op = op;         // a label, only used for drawing
    if (TAPE) TAPE.push(this);
  }
}

const add  = (a, b) => new Value(a.data + b.data, [a, b], [1, 1], "+");
const mul  = (a, b) => new Value(a.data * b.data, [a, b], [b.data, a.data], "×");
const mulc = (a, c) => new Value(a.data * c, [a], [c], "×"); // times a fixed number

function tanh(a) {
  const t = Math.tanh(a.data);
  return new Value(t, [a], [1 - t * t], "tanh");
}
const relu = (a) => new Value(Math.max(0, a.data), [a], [a.data > 0 ? 1 : 0], "ReLU");
function sigmoid(a) {
  const s = 1 / (1 + Math.exp(-a.data));
  return new Value(s, [a], [s * (1 - s)], "σ");
}
const neglog = (a) => new Value(-Math.log(a.data), [a], [-1 / a.data], "−log");
const square = (a) => new Value(a.data * a.data, [a], [2 * a.data], "²");
// σ and the loss fused into one step, as chapters 1 and 2 train.
// For a dot labeled y, its slope simplifies to σ(z) − y.
function bceWithLogits(z, y) {
  const s = 1 / (1 + Math.exp(-z.data));
  const loss = Math.max(z.data, 0) - z.data * y + Math.log1p(Math.exp(-Math.abs(z.data)));
  return new Value(loss, [z], [s - y], "loss");
}

function backward(loss) {
  loss.grad = 1;
  for (let k = TAPE.length - 1; k >= 0; k--) {
    const v = TAPE[k];
    if (!v.inputs) continue;
    for (let i = 0; i < v.inputs.length; i++) {
      v.inputs[i].grad += v.local[i] * v.grad; // the chain rule
    }
  }
}
