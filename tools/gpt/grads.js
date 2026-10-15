// Secret 5: how much gradient reaches each block's weights, at the same step of the GPT's run and of the two variants
// (without residual connections, without layer norms), measured as package.js measures it (L.blockGrads).
// node grads.js [step=3000]
const L = require("./lib.js");
const { step } = L.args({ step: 3000 });
for (const [label, run, code] of [["GPT", "R", L.GPT], ["without residual connections", "Rnr", L.variant("nores")], ["without layer norms", "Rnn", L.variant("nonorm")]]) {
  const g = L.blockGrads(code, L.loadModel(`${run}_${step}.bin`));
  console.log(`${label.padEnd(30)} step ${step}: ${g.map((x, l) => `block ${l + 1} ${x}`).join(", ")} | block 1 / block ${g.length}: ${(g[0] / g[g.length - 1]).toPrecision(3)}`);
}
