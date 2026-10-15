// The page runs chapter 7's network with a compact forward pass of its own (forward7 in gpt/index.html), and chapter
// 6's with probs6; check both against those chapters' own code, with their shipped weights, on test stories: the same
// probabilities, from the start and from a later position. Also checks that the page's tokenizer is chapter 6's.
// node check7.js
const L = require("./lib.js");
const main = L.html.match(/<script>\n([\s\S]*?)\n<\/script>/g).pop();
const grab = (name) => { const m = main.match(new RegExp(`  function ${name}\\(([\\s\\S]*?)\\n  }\\n`)); if (!m) throw new Error("no " + name); return `function ${name}(${m[1]}\n}`; };
const forward7 = new Function(grab("forward7") + "\nreturn forward7;")();
const m7 = L.A7.loadShipped(), test = L.storiesOf("test.bin").stories;
let worst = 0, n = 0;
for (const s of [0, 1, 2, 3, 57]) {
  const { tokens } = test[s];
  for (const from of [0, Math.floor(tokens.length / 2), tokens.length - 1]) {
    const a = forward7(m7, tokens, from).p, b = L.ATN.forward(m7, tokens, from).p;
    for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
    n++;
  }
}
console.log(`forward7 against chapter 7's forward, ${n} passes: largest probability difference ${worst.toExponential(1)}`);
console.log("tokenizer script identical to chapter 6's:", L.script("tok") === L.W6.script("tok"));
// chapter 6's network is run with the same probs6 and ctx6 as chapter 7's page, whose numbers chapter 7's tools checked
const main7 = L.A7.html.match(/<script>\n([\s\S]*?)\n<\/script>/g).pop(), line = (src, name) => (src.match(new RegExp(`\\n  (?:function ${name}\\(|const ${name} = )[^\\n]*(?:\\n    [^\\n]*)*`)) || [""])[0];
console.log("probs6 and ctx6 identical to chapter 7's page:", ["probs6", "ctx6"].every((f) => line(main, f) && line(main, f) === line(main7, f)));
// the page's rules for names (nameKit, mentionedAt, heroOf, swapName) and made-up names for its own trainer (makeName,
// applyRename) against rename.js, which trained the GPT and makes lib.js's test stories
{
  const RN = L.RN, tk = L.tokenizer(), V = tk.vocab, invent = L.INVENT, banned = new Set(), ids = L.names();
  for (const w of invent) { const Z = tk.encode(w); for (let j = 1; j < Z.length; j++) banned.add(Z[j - 1] * 2000 + Z[j]); }
  const page = new Function("mulberry32", ["makeName", "nameKit", "mentionedAt", "heroOf", "swapName", "applyRename"].map(grab).join("\n") + "\nreturn { makeName, nameKit, heroOf, swapName, applyRename };")(L.mulberry32);
  const k = page.nameKit(V), same = (a, b) => a.length === b.length && a.every((x, j) => x === b[j]);
  console.log("the page's nameKit equals rename.js's:", same(Array.from(k.glued), Array.from(RN.kit().glued)) && same(Array.from(k.owner), Array.from(RN.kit().owner)) && k.s === RN.kit().s, `| names with a possessive of their own: ${ids.filter((x) => k.owner.includes(x)).map((x) => V[x] + "'s").join(" ")}`);
  // every training story, renamed with chance one half, and renamed for sure: the page against rename.js; and after
  // renaming for sure, no mention of the hero as a word may be left
  const { stories } = L.storiesOf("train.bin");
  let equal = 0, renamed = 0, heroes = 0, left = 0, poss = 0, longer = 0;
  for (const p of [0.5, 1]) {
    const r1 = L.mulberry32(11), r2 = L.mulberry32(11);
    for (const st of stories) {
      const d1 = RN.draw(r1), d2 = [r2(), Math.floor(r2() * 4294967296)];
      const a = RN.applyRename(st, p, d1), b = page.applyRename(st, p, d2, ids, tk.encode, banned, invent, k);
      if (same(a.tokens, b.tokens) && same(a.next, b.next)) equal++;
      if (p === 1) {
        const X = RN.hero(st.tokens);
        if (page.heroOf(st.tokens, ids, k) !== X) left += 1e6; // the two must pick the same hero
        if (X < 0) continue;
        heroes++; if (a !== st) renamed++;
        for (let i = 0; i < a.tokens.length; i++) if (RN.mentionedAt(a.tokens, i, X, RN.kit())) left++;
        if (st.tokens.some((x) => k.owner[x] === X)) poss++;
        if (st.tokens.some((x, j) => x === X && j + 1 < st.tokens.length && k.glued[st.tokens[j + 1]])) longer++;
      }
    }
  }
  console.log(`made-up names: the page's renaming equals rename.js on ${equal} of ${2 * stories.length} (every training story, renamed with chance one half and for sure); ${heroes} stories have a hero, ${renamed} renamed for sure, with ${left} mentions of the hero left; ${poss} of them have the hero's possessive as one token, ${longer} the hero's token beginning a longer word`);
  // the page's test stories: the page's swapName against lib.js's invented, for each of the first 100 test stories'
  // hero and each invented name
  const test = L.storiesOf("test.bin").stories.slice(0, 100); let tEq = 0, tN = 0;
  for (const { tokens } of test) { const X = RN.hero(tokens); if (X < 0) continue; for (const w of invent) { const Z = tk.encode(w), a = L.invented(tokens, X, Z), b = page.swapName(tokens, X, Z, k); tN++; if (same(Array.from(a.toks), b.out) && same(a.ment, b.at)) tEq++; } }
  console.log(`invented names in the test stories: the page's swapName equals lib.js's invented in ${tEq} of ${tN} (story, name) pairs`);
  // two examples: a hero's possessive becomes the made-up name and "##'s"; a longer word that begins with the hero's
  // token stays
  const after = (st) => tk.decode(Array.from(RN.applyRename(st, 1, [0, 7]).tokens)), tid = tk.id.get("tom");
  const withPoss = after(stories.find((st) => RN.hero(st.tokens) === tid && st.tokens.includes(tk.id.get("tom's"))));
  const withLonger = after(stories.find((st) => RN.hero(st.tokens) === tid && /tomato|tomorrow/.test(tk.decode(Array.from(st.tokens)))));
  console.log(`a story with "tom's": after renaming, "tom" and "tom's" ${/\b(tom|tom's)\b/.test(withPoss) ? "REMAIN" : "are gone"}, "##'s" follows the new name ${/[a-z] ?'s\b/.test(withPoss) ? "yes" : "NO"}; a story with "tomato" or "tomorrow": "tom" ${/\btom\b/.test(withLonger) ? "REMAINS" : "is gone"}, the longer word ${/tomato|tomorrow/.test(withLonger) ? "stays" : "IS BROKEN"}`);
}
