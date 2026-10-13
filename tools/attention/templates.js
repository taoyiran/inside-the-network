// Candidate stories for the page's question and film: a name in the first sentence, and an ending where the name
// should come back from far away. For each: the attention network's and chapter 6's probability and rank of the name,
// for several names (one-token names, a rarer one, a two-piece one).
// node templates.js F8_19000.bin
const L = require("./lib.js");
const tk = L.tokenizer(), m = L.loadModel(process.argv[2]), m6 = L.W6.loadModel("Q128_600000.bin");
const NAMES = ["lily", "tom", "sue", "max", "fred", "mia", "nora", "zarko"];
const T = [
  "once upon a time, there was a little girl named {}. she loved to play in the park with her red ball. one day, the ball rolled into a big puddle. a kind dog jumped in and brought the ball back. the dog wagged its tail and gave the ball to",
  "once upon a time, there was a little girl named {}. she loved to play in the park with her red ball. one day, the ball rolled into a big puddle. a kind dog jumped in and brought the ball back.",
  "once upon a time, there was a little girl named {}. she loved to play in the park with her red ball. one day, the ball rolled into a big puddle. a kind dog jumped in and brought the ball back. “thank you, dog!” said",
  "once upon a time, there was a little girl named {}. she had a big red ball. one day, she went to the park to play. the sun was warm and the birds were singing. she kicked the ball high into the sky.",
  "once upon a time, there was a boy named {}. he lived in a small house with his mom and dad. every morning, he went outside to look at the flowers. one day, he saw a little bird in the garden. the bird looked at him and said, “hello,",
  "once upon a time, there was a girl named {}. she had a cat. the cat was very soft and liked to sleep in the sun. one day, the cat went missing. everyone looked for the cat in the house and in the garden. at last,",
  "there was a little girl named {}. she liked to draw pictures of the sea, the sun and the trees. one day, her mom gave her a big box of new crayons.",
  "once upon a time, in a small town, there lived a girl named {}. she had a best friend, a little white dog. they played together every day in the big green park near the river. one day, the dog found a shiny stone. he brought it to",
];
const encode = (s) => [0, ...tk.encode(s)];
const rank = (p, x) => { let r = 1; for (let t = 0; t < p.length; t++) if (p[t] > p[x]) r++; return r; };
T.forEach((tpl, i) => {
  const row = NAMES.map((nm) => {
    const toks = Uint16Array.from(encode(tpl.replace("{}", nm))), n = toks.length, want = tk.encode(nm)[0];
    const pa = L.ATN.forward(m, toks, n - 1).p, c = Array.from({ length: 8 }, (_, j) => (n - 8 + j >= 1 ? toks[n - 8 + j] : 0)), p6 = L.NWM.forward(m6, c).p;
    let top = 0; for (let t = 1; t < pa.length; t++) if (pa[t] > pa[top]) top = t;
    return `${nm} ${(100 * pa[want]).toFixed(0)}%#${rank(pa, want)} (ch6 ${(100 * p6[want]).toFixed(0)}%#${rank(p6, want)}) top:${tk.vocab[top]}`;
  });
  console.log(`[${i}] …${tpl.slice(-40)}\n   ${row.join(" | ")}`);
});
