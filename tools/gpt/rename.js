// Names, for training and for the page's test. TinyStories has few names: about two dozen one-token names, such as
// "lily" and "tom", carry most of its stories, so a network trained on it seldom has to copy a name it does not know.
// With rename=p, a training story whose hero has one of the page's one-token names gets, with chance p, a made-up name
// in its place, everywhere the hero is mentioned as a word: its token standing alone, before an apostrophe ending
// ("##'s"), or as its possessive when the vocabulary has one as a single token ("tom's", which becomes the made-up
// name's pieces and "##'s"); not where its token begins a longer word, as "tom" does "tomato". The made-up name is 2
// or 3 random syllables cut into pieces by the tokenizer, never one of the 6 names the page tests with, never
// containing a pair of their pieces, never a single token. The page's test gives its name stories the 6 invented names
// by the same rule (invented in lib.js). nameKit, mentionedAt, heroOf and swapName are copies of the page's own
// (gpt/index.html); check7.js checks that the two give the same stories.
// This module needs only chapter 7's library, not chapter 8's page, so that chapter 7's trainers can use it too.
const A7 = require("../attention/lib.js");
const INVENT = ["zarko", "vorna", "quilla", "morvin", "dashu", "tilda"];
const CONS = "b c d f g h j k l m n p r s t v w z".split(" "), VOW = "a e i o u".split(" "), ENDS = ["", "", "", "n", "r", "l", "s", "x"];
// what the rules need from the vocabulary: glued[t] is 1 for the tokens that continue the word of the token before
// ("##…"), apart from apostrophe endings ("##'s"); owner[t], for a name's possessive that is a single token ("tom's"),
// is the name's token ("tom"), and -1 for every other token; s is the token "##'s"
function nameKit(vocab) {
  const id = new Map(vocab.map((t, i) => [t, i])), owner = new Int32Array(vocab.length).fill(-1);
  vocab.forEach((t, i) => { const q = id.get(t + "'s"); if (q !== undefined && !t.startsWith("##")) owner[q] = i; });
  return { glued: Uint8Array.from(vocab, (t) => (t.startsWith("##") && !t.startsWith("##'") ? 1 : 0)), owner, s: id.get("##'s") };
}
// whether the name x is mentioned as a word at position i of t: its token, not followed by a piece that continues it,
// or its single-token possessive
function mentionedAt(t, i, x, k) {
  return (t[i] === x && !(i + 1 < t.length && k.glued[t[i + 1]])) || k.owner[t[i]] === x;
}
// a story's hero: of the name tokens ids, the one mentioned most often as a word (ties: the lower token); -1 if none
function heroOf(t, ids, k) {
  const cnt = new Map();
  for (let i = 0; i < t.length; i++) {
    const x = k.owner[t[i]] >= 0 ? k.owner[t[i]] : t[i];
    if (ids.includes(x) && mentionedAt(t, i, x, k)) cnt.set(x, (cnt.get(x) || 0) + 1);
  }
  return cnt.size ? [...cnt].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0] : -1;
}
// the tokens t with every mention of the name x as a word replaced by the pieces Z (its single-token possessive by Z
// and "##'s"); t[0], the <end> that opens a story, is kept; and where each mention's pieces start
function swapName(t, x, Z, k) {
  const out = [t[0]], at = [];
  for (let i = 1; i < t.length; i++) {
    if (mentionedAt(t, i, x, k)) { at.push(out.length); out.push(...Z); if (t[i] !== x) out.push(k.s); } else out.push(t[i]);
  }
  return { out, at };
}
let SETUP = null;
function setup() {
  if (SETUP) return SETUP;
  const tk = A7.tokenizer(), banned = new Set();
  for (const w of INVENT) { const Z = tk.encode(w); for (let j = 1; j < Z.length; j++) banned.add(Z[j - 1] * 2000 + Z[j]); }
  return (SETUP = { tk, banned, kit: nameKit(tk.vocab) });
}
const kit = () => setup().kit;
// a story's hero, among the page's one-token names
const hero = (tokens) => heroOf(tokens, A7.names(), kit());
function makeName(r) {
  const { tk, banned } = setup();
  for (;;) {
    let s = ""; const syl = 2 + Math.floor(r() * 2);
    for (let j = 0; j < syl; j++) s += CONS[Math.floor(r() * CONS.length)] + VOW[Math.floor(r() * VOW.length)];
    s += ENDS[Math.floor(r() * ENDS.length)];
    const Z = tk.encode(s);
    if (Z.length < 2 || INVENT.includes(s)) continue;
    let ok = true; for (let j = 1; j < Z.length; j++) if (banned.has(Z[j - 1] * 2000 + Z[j])) ok = false;
    if (ok) return Z;
  }
}
// For every story of a step, the trainer draws two numbers from the run's random stream, whether or not the story is
// renamed, so that renaming never shifts the rest of the run: a roll (renamed if under p) and a seed for the name.
const draw = (r) => [r(), Math.floor(r() * 4294967296)];
// a training story { tokens, next } with its hero renamed, or the story as it is
function applyRename(story, p, [roll, seed]) {
  if (!(p > 0) || roll >= p) return story;
  const X = hero(story.tokens);
  if (X < 0) return story;
  const all = Uint16Array.from([...swapName(story.tokens, X, makeName(A7.mulberry32(seed)), kit()).out, 0]); // the story, then the <end> that closes it
  return { tokens: all.subarray(0, all.length - 1), next: all.subarray(1) };
}
module.exports = { INVENT, nameKit, mentionedAt, heroOf, swapName, kit, hero, makeName, draw, applyRename };
