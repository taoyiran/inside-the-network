# How Chapter 7's network is trained

A complete account of how the attention network in [Chapter 7, Attention](https://taoyiran.github.io/inside-the-network/attention/) was built, trained, measured and shipped, as of 13 October 2026. Everything here can be rerun with the scripts in [`tools/attention/`](../tools/attention/); the last section gives the commands.

## Overview

Chapter 7's network has about as many weights as chapter 6's, but reads up to 256 tokens back instead of 8. It trained for 19,000 steps of 8 whole stories on 16 processor cores, in 33 minutes. On 100 stories it never saw, its first guess is right 46.6% of the time, against 41.8% for chapter 6's network and for counting. The gain is largest where the answer appeared earlier in the story, out of chapter 6's reach: when the answer is the story's name, 45.3% against 2.6%.

Everything below runs on the page's own code: the network in `<script id="attn">` of [`attention/index.html`](../attention/index.html), with chapter 6's tokenizer (the page carries an identical copy of `<script id="tok">`). The build scripts read that code straight out of the pages.

| Item | Value |
| --- | --- |
| Training text | chapter 6's 20,000 TinyStories stories, 4,300,373 tokens |
| Vocabulary | chapter 6's 2,000 tokens |
| Network | embedding table → 8 attention heads over up to 256 tokens → 128 tanh neurons → 2,000 scores → softmax |
| Parameters | 352,848 (chapter 6: 354,896) |
| Loss | cross-entropy, averaged over every position of every story in the step |
| Optimizer | Adam (β₁ 0.9, β₂ 0.999, ε 10⁻⁸), learning rate 0.008, falling linearly to 0 over the last 30% of steps |
| Batch | 8 whole stories, each picked by a random training token (2,066 positions per step on average) |
| Length | 19,000 steps = 39,246,369 positions, about 9.1 passes over the training tokens |
| Randomness | one seed (1) sets the starting weights and every batch |
| Hardware and time | 16 worker threads on an Apple M5 Pro (6 performance and 12 efficiency cores), 1,981 seconds |
| Validation, stories 1,001–1,300 (64,301 tokens) | loss 2.389 nats (perplexity 10.9), 47.3% right first guesses; chapter 6's network: 2.736, 42.1% |
| Test, first 100 stories (21,730 tokens) | 46.6% right first guesses; chapter 6's network and counting with the last 4 tokens: 41.8% |
| Test, all 1,000 stories (211,846 tokens) | 47.3%; chapter 6's network 42.2%, counting 42.1% |

## The data

The same files and splits as [chapter 6](chapter-6-training.md#the-data): training stories 2 to 20,001 of `TinyStoriesV2-GPT4-train.txt`, the first 1,000 complete stories of the validation file as test stories (the page measures on the first 100), and validation stories 1,001 onward for choosing settings. Chapter 7 ships no data of its own; the page loads `words/stories.js`.

**A story as the network reads it.** Each story is read on its own, starting with the `<end>` token (number 0) that closes the story before it, so every story opens the same way. A story of L tokens gives L + 1 positions: position 0 reads `<end>` and guesses the story's first token, and the last position guesses the `<end>` that closes the story. Position i can read the tokens at i − 255 to i of its own story, never a later token and never another story. Training stories average 215 tokens; 84% fit in the 256-token window whole, and for longer ones each position reads its last 256 tokens.

## The network

| Part | Shape | Numbers | Starting values |
| --- | --- | --- | --- |
| Embedding table `E` | 2,000 rows × 32 | 64,000 | 0.1 × a standard normal |
| Queries `Wq` | 8 heads × 8 × 32 | 2,048 | a standard normal ÷ √32 |
| Keys `Wk` | 8 heads × 8 × 32 | 2,048 | a standard normal ÷ √32 |
| Values `Wv` | 8 heads × 16 × 32 | 4,096 | a standard normal ÷ √32 |
| Distance scores `R` | 8 heads × 256 | 2,048 | 0 |
| Hidden weights `W1` | 128 × 160 | 20,480 | a standard normal ÷ √160 |
| Hidden biases `b1` | 128 | 128 | 0 |
| Output weights `W2` | 2,000 × 128 | 256,000 | a standard normal ÷ √128 |
| Output biases `b2` | 2,000 | 2,000 | 0 |
| Total | | 352,848 | |

Normal draws use the Box–Muller formula on the seeded stream, filling `E`, `Wq`, `Wk`, `Wv`, `W1`, `W2` in that order, as in chapter 6.

**Where order comes from.** A dot product of a query and a key does not depend on where the key is, so each head learns a score for every distance back, `R[h][i − j]`, added to the dot product before softmax: a learned relative-position bias, in the spirit of T5's relative position buckets and ALiBi. GPT-2 instead adds a learned row per absolute position to the token's row. Relative scores suit stories longer than the window (a position always reads the last 256 tokens, wherever it is), and each head's curve can be read straight off the weights.

**No residual connection, no layer norm.** The hidden layer reads the token's own row and the 8 heads' results side by side; there is a single attention layer. Residual connections, layer norm and stacking are chapter 8's.

## The forward pass

For position i of a story with tokens t₀ = `<end>`, t₁, …, and each head h:

```math
q_i = W_q^{(h)} E_{t_i}, \qquad k_j = W_k^{(h)} E_{t_j}, \qquad v_j = W_v^{(h)} E_{t_j}
```

```math
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{8}} + R^{(h)}_{i-j}, \qquad w_{ij} = \frac{e^{\,s_{ij} - \max_{j'} s_{ij'}}}{\sum_{j'} e^{\,s_{ij'} - \max_{j'} s_{ij'}}}, \qquad o^{(h)}_i = \sum_{j} w_{ij}\, v_j, \qquad \max(0, i-255) \le j \le i
```

```math
x_i = \big[\,E_{t_i};\ o^{(1)}_i;\ \dots;\ o^{(8)}_i\,\big] \in \mathbb{R}^{160}, \qquad a = \tanh(W_1 x_i + b_1), \qquad p = \operatorname{softmax}(W_2 a + b_2)
```

The loss at position i is −ln p(t_{i+1}) (the `<end>` closing the story, at the last position). A step's loss is the mean over all positions of its 8 stories.

**Cost.** For a story of n tokens, the queries, keys and values take about n × 8 × 32 × 32 multiply-adds, the scores and weighted sums about n × min(n, 256) / 2 × 8 × 24, and the hidden and output layers n × (160 × 128 + 128 × 2,000). At these sizes the output layer dominates: for a 215-token story, about 59 million of the 66 million multiply-adds.

## The backward pass

The output and hidden layers are chapter 6's, down to the 160 numbers that were read: the first 32 go to the token's own row, 16 to each head's result. Then, per head:

```math
\frac{\partial L}{\partial v_j} \mathrel{+}= w_{ij}\, \frac{\partial L}{\partial o_i}, \qquad \frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial o_i} \cdot v_j, \qquad \frac{\partial L}{\partial s_{ij}} = w_{ij}\Big(\frac{\partial L}{\partial w_{ij}} - \sum_{j'} w_{ij'} \frac{\partial L}{\partial w_{ij'}}\Big)
```

```math
\frac{\partial L}{\partial q_i} \mathrel{+}= \frac{\partial L}{\partial s_{ij}} \frac{k_j}{\sqrt 8}, \qquad \frac{\partial L}{\partial k_j} \mathrel{+}= \frac{\partial L}{\partial s_{ij}} \frac{q_i}{\sqrt 8}, \qquad \frac{\partial L}{\partial R_{i-j}} \mathrel{+}= \frac{\partial L}{\partial s_{ij}}
```

Last, the queries, keys and values send their gradients to `Wq`, `Wk`, `Wv` and to the rows of `E` they were made from. A token's row therefore learns from three places: as the token being read at its own position, and as a key and a value for every later position that reads it.

## Adam and the schedule

Chapter 6's Adam, applied to the step's gradient averaged over its positions. The rate stays at 0.008 for the first 13,300 steps, then falls linearly to 0.008 ÷ 5,700 at the last step. No weight decay, gradient clipping or dropout.

## Training on several processor cores

`tools/attention/ptrain.js` draws the 8 stories exactly as the page's trainer does, then cuts all their positions into one run and gives each of 16 worker threads an equal share. A worker's share can start in the middle of a story: `forward(m, tokens, from)` computes the queries, keys and values for the 255 positions before `from` as well, but outputs, losses and gradients only for positions `from` onward. Gradients add up over positions, so the split changes nothing but the order of the additions. Each worker then sums all the workers' gradients for its sixteenth of the weights and applies the Adam update to it.

With 1 worker the result is bit-identical to the page's one-thread `trainStep` (`train.js`, checked after 20 steps). The run made about 9.6 steps a second, about 19,800 positions a second.

## How the settings were chosen

Each candidate trained for 2,000 steps (about 4.1 million positions, with the same schedule shape), measured on all tokens of validation stories 1,001 to 1,300 (`sweep.sh`). After about as many examples, chapter 6's network was at about 38%, on its own validation sample.

| Question | Compared | Validation accuracy after ~4.1M positions | Kept |
| --- | --- | --- | --- |
| Learning rate (8 heads) | 0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016 | 29.2, 36.0, 40.9, 42.2, 43.0, 42.9, 42.8% | 0.008, the lower end of the plateau, for a 10 times longer run |
| Heads (values 128 numbers in all) | 4 heads of 32, 8 of 16, 16 of 8, at 0.002 | 34.6, 36.0, 35.8% | 8 |
| Heads at 0.006 | 8 of 16 against 16 of 8 | 42.2 against 42.5% | 8, for fewer cards to read |
| Batch | 8 stories at 0.008 against 4 stories at 0.004 for twice the steps | 43.0 against 42.5% | 8 stories |
| Window | 256 against 16 tokens, at 0.006 | 42.2 against 40.8% | 256 |

The 16-token window also learned names far less. After 2,000 steps, the story's name, last mentioned more than 8 tokens back, was the first guess 3.0% of the time with a 16-token window (learning rate 0.002) and 37.6% with the 256-token window (learning rate 0.004), over the 1,000 test stories.

## The final run and its results

The validation accuracy over the run (stories 1,001–1,300): 18.9% after 50 steps, 38.1% after 1,000, 41.6% after 3,000, 43.6% after 10,000, then 47.3% at the end, most of the last gain coming while the rate fell.

On the page's 21,730 test tokens (every token of the first 100 test stories):

| Where the right answer last appeared | Share | Counting, last 4 | Chapter 6 | Attention |
| --- | --- | --- | --- | --- |
| All tokens | 100% | 41.8% | 41.8% | 46.6% |
| Among the last 8 tokens | 9.6% | 55.1% | 55.6% | 60.1% |
| Earlier in the story, further back | 46.2% | 42.8% | 44.2% | 52.6% |
| … and not one of the 100 most common tokens | 11.2% | 21.1% | 20.0% | 36.8% |
| … and the story's name | 1.6% | 3.5% | 2.6% | 45.3% |
| Nowhere earlier in the story | 44.2% | 37.8% | 36.4% | 37.4% |

"The story's name" is the one-token name (from a list of 25 such as "tim", "lily", "tom") mentioned most often in the story, at places where its last mention is more than 8 tokens back. Over all 1,000 test stories (3,443 such places in 660 stories): attention 46.4%, chapter 6 3.9%, counting 5.6%.

**What the heads do** (weight by distance back, over every position of the 100 test stories): heads 1 to 3 read mostly the last few tokens, head 3 most sharply (41% of its weight on the token just before); head 4 splits between the token just before (31%) and far back; heads 5 to 8 read mostly 8 or more tokens back. Head 6 reads names (its most-read tokens per appearance, after `<end>`: "sara", "leo", "mia", "lily"), head 7 nouns such as "baby", "people", "fish", and heads 5 and 8 the opening of the story. `<end>` tops the list of 7 of the 8 heads, because every later position can see it; head 8 puts about 11% of all its weight there.

**Switching heads off** (a head's value table set to 0), on the same tokens:

| Head off | All | Last 8 | Further back | The name | New |
| --- | --- | --- | --- | --- | --- |
| none | 46.6% | 60.1% | 52.6% | 45.3% | 37.4% |
| 1 | 43.4% | 56.9% | 49.2% | 59.7% | 34.3% |
| 2 | 41.3% | 56.9% | 45.8% | 30.9% | 33.2% |
| 3 | 39.2% | 53.9% | 45.8% | 52.6% | 29.1% |
| 4 | 43.4% | 56.9% | 50.0% | 42.1% | 33.5% |
| 5 | 41.4% | 51.9% | 49.1% | 37.1% | 31.2% |
| 6 | 45.1% | 58.4% | 50.0% | 11.5% | 37.1% |
| 7 | 45.0% | 56.6% | 50.3% | 36.2% | 37.0% |
| 8 | 43.9% | 56.4% | 49.3% | 38.8% | 35.4% |
| all distance scores set to 0 | 26.8% | 44.6% | 31.2% | 37.4% | 18.3% |

**Swapping the name.** In the 63 test stories (of the first 100) whose name comes back from further than 8 tokens, the name was replaced everywhere by another one-token name. At those 340 places the new name became the first guess: "tom" 50.6%, "max" about 49%, "sue" about 46%, "mia" about 41%, "lily" 40.3%, "ben" about 35%, "fred" 21.8% ("fred" appears 301 times in the training stories, the others thousands of times). The old name stayed the first guess about 1% of the time. Over the 1,000 test stories, with each story's name swapped for another, 32.2% against 46.4% for the stories' own names. The per-name rates moved by several points between checkpoints 17,000, 18,000 and 19,000; "fred" was the lowest at all three.

**Invented names.** In the 63 name stories, the hero was given each of 6 made-up names of 3 pieces in turn ("z ##ark ##o", "qu ##ill ##a", …; `pieces.js`); each name then has 63 first mentions and 386 later ones. At later mentions:

- the piece right after the first ("##ark" after "z") was never the first guess, for any of the six names (0 of 386 each). None of these pairs appears in the training stories (or, for "qu ##ill", twice).
- the third piece, at later mentions, was guessed 300 of 386 times for "qu ##ill ##a" (0 of 63 at the first mention) and 293 of 386 for "mor ##v ##in" (21 of 63 at the first mention). For "t ##ild ##a" it went the other way: 39 of 63 at the first mention, 30 of 386 later. The other three were never guessed. "##ill ##a" appears 79 times in the training stories ("gorilla", "vanilla"), "##v ##in" 16 times and "##ild ##a" 5 times; the other pairs never.

So the network does use the story, on the whole: a piece that already appeared in it and that can follow the piece before (the pair occurs in training) often becomes much more likely, though not always. What one layer cannot do is find the earlier "z" and read the token after it: every key is made from its own token's row alone, so no key can carry "the token before me". A previous-token head in one layer and an induction head in the next can (Elhage et al. 2021, Olsson et al. 2022); that is chapter 8. Over the 1,000 test stories, counting every piece and only later mentions more than 8 tokens after the previous one, the next piece was the first guess 3.3% of the time at the first mention and 8.2% at later mentions (`names.js`).

**Not kept for the page.** Two other ideas were measured and dropped. The weight heads put on the `<end>` token that opens the story (a possible "attention sink"), over the first 255 positions of the 100 test stories, was at most 11.8% (head 8), against 2.5% if every head spread its weight evenly: there, but too weak to be a secret of its own. At "she", "he", "her" and "his", no head put much weight on the story's name (at most 4.7% against an even share of 3.5%): in one layer the position reading a pronoun is busy guessing the next word, usually a verb.

## From checkpoint to page

`tools/attention/package.js` writes `attention/weights.js` (1,888,168 bytes): the network's 352,848 numbers rounded to 32-bit floats (every number the page shows is measured with these), the run's validation log, and the saved moments for Secret 1. A moment holds, for the checkpoints saved at steps 50, 100, 200, 500, 1,000, 3,000 and 10,000 of the run, and for the starting weights (step 0), each head's share of weight in 8 distance bins and its 4 most-read tokens, measured on the page's 100 test stories exactly as the page measures the finished network. Step 0 is the starting weights, recomputed from seed 1.

## Training in the reader's browser

The page's "Train" button runs the same code in a Web Worker: a fresh network from seed 2 (each Reset moves to the next seed), 8 whole stories per step picked as above, at a constant learning rate of 0.008. In headless Chrome on an Apple M5 Pro it made about one step a second (about 2,000 positions a second), so a reader's network sees in a minute what the finished run saw in its first 60 steps. Every 60 seconds the page measures it on the first 20 test stories. Measurements over all 100 test stories (the ladder, the heads, the swaps, the knock-outs, the invented names) run in a second worker.

## Checks

| Check | How | Result |
| --- | --- | --- |
| Backward pass | `gradcheck.js`: central differences on small networks, every weight array, windows shorter than the story and outputs from a later position | largest relative difference 2.3 × 10⁻⁶ (h = 10⁻⁵) |
| Backward pass, in the page | at every load, 45 weights from every part of the finished network against central differences | largest relative difference about 10⁻⁶ |
| Python translation | `pycheck.js`: the page's Python against its JavaScript on the real shapes | probabilities within 4 × 10⁻¹⁹, losses identical to 9 decimals, weights after 3 Adam steps within 1.7 × 10⁻¹² |
| Multi-core trainer | `ptrain.js` with 1 worker against `train.js` (the page's `trainStep`), 20 steps | bit-identical |
| Packaging | `verify.js`: `weights.js` against the checkpoint; every saved moment against a fresh measurement of its checkpoint; the page's tokenizer against chapter 6's | equal up to 32-bit rounding; identical; identical |
| Live against offline | the page's numbers in headless Chrome against `compare.js`, `ablate.js` and the swap measurements | the same |
| Reviews | an independent accuracy review of the page and a code review | findings fixed before publishing |

## How to reproduce it

The commands run from `tools/attention/` and need Node 24 or later and chapter 6's token files (run chapter 6's `fetch.sh` and `prep.js` first, see [chapter 6's write-up](chapter-6-training.md#how-to-reproduce-it)); `pycheck.js` also needs Python 3. Checkpoints and logs go to `tools/attention/work/`, which git ignores.

```sh
cd tools/attention
mkdir -p work/runs
node ptrain.js lr=0.008 steps=19000 save=F8 > work/runs/F8.txt
                              # the finished network: about 33 minutes with 16 workers on an Apple M5 Pro
node package.js F8 19000      # writes attention/weights.js
node verify.js F8 19000       # weights.js against the checkpoint; the tokenizer against chapter 6's
node gradcheck.js             # the backward pass against central differences
node pycheck.js               # the Python translation against the JavaScript
node compare.js F8_19000.bin  # counting, chapter 6 and attention, by where the answer last appeared
node names.js F8_19000.bin    # names far back, swapped names and invented names, over the 1,000 test stories
node ablate.js F8_19000.bin   # each head switched off (heads=R: every distance score set to 0)
node heads.js F8_19000.bin    # where each head's weight goes
node swapnames.js F8_19000.bin   # Secret 2 per new name (also on F8_17000.bin, F8_18000.bin)
node pieces.js F8_19000.bin   # Secret 4 piece by piece, with how often each pair appears in training
sh sweep.sh                   # the settings sweep, 13 short runs, about an hour
```

`node train.js` trains with the page's own `trainStep` on one thread. `templates.js` scores candidate stories for the page's question, `pronouns.js` is the dropped pronoun measurement, `look.js` and `interact.js` open the served page in headless Chrome (serve the repo with `python3 -m http.server 8765`) and print its live readouts and the results of clicking through its controls.
