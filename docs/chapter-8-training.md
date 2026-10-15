# How Chapter 8's GPT is trained

A complete account of how the GPT in [Chapter 8, A tiny GPT](https://taoyiran.github.io/inside-the-network/gpt/) was built, trained, measured and shipped, as of 15 October 2026. Everything here can be rerun with the scripts in [`tools/gpt/`](../tools/gpt/); the last section gives the commands.

## Overview

Chapter 8's GPT has 4 blocks of attention and an MLP, with residual connections and layer norms: 461,136 numbers, 1.3 times chapter 7's network. It trained for 9,500 steps of 16 whole stories (40.3 million positions, about 9.4 passes over the training tokens) with 16 worker threads. On the page's 100 test stories, its first guess is right 52.1% of the time, against 46.6% for chapter 7's network and 41.8% for chapter 6's and for counting.

What only a stack can do shows in invented names. At later mentions of an invented name, for pieces that never follow the piece before them in training (such as "##or" after "v" in "v ##or ##na"), the GPT's first guess is the right next piece 61.4% of the time (the page's measure), against 0.0% for chapter 7's network. Two heads do it, a previous-token head in block 1 and an induction head in block 2; switching off either one drops the rate to 0.0% or 4.6%.

It learned this only because, each time a training story whose hero has a common one-token name was drawn, the hero was given a made-up name with chance one half. Trained for as many steps on the stories as they are, the same network did nearly as well over all tokens (51.9% against 52.1%) but finished only 5.4% of those pieces: TinyStories seldom asks a network to copy a name it does not know.

| Item | Value |
| --- | --- |
| Training text | chapter 6's 20,000 TinyStories stories, 4,300,373 tokens; each story whose hero has a listed one-token name given a made-up name with chance one half, each time it is drawn |
| Vocabulary | chapter 6's 2,000 tokens |
| Network | embedding table → 4 blocks (layer norm, 4 heads over up to 256 tokens, W<sub>o</sub>, add; layer norm, an MLP of 256 GELU neurons, add) → layer norm → 2,000 scores → softmax |
| Parameters | 461,136 (chapter 7: 352,848; chapter 6: 354,896) |
| Loss | cross-entropy, averaged over every position of every story in the step |
| Optimizer | Adam (β₁ 0.9, β₂ 0.999, ε 10⁻⁸), learning rate 0.004, falling linearly to 0 over the last 30% of steps |
| Batch | 16 whole stories, each picked by a random training token (4,243 positions per step on average) |
| Length | 9,500 steps = 40,312,903 positions, about 9.4 passes over the training tokens |
| Randomness | one seed (1) sets the starting weights, every batch and every made-up name |
| Hardware and time | 16 worker threads on an Apple M5 Pro (6 performance and 12 efficiency cores): 6,544 seconds, sharing the machine for its first 4,149 seconds with the retraining of chapter 7's network (8 more threads); the run without made-up names, alone, took 5,633 seconds |
| Validation, stories 1,001–1,300 (64,301 tokens) | loss 2.090 nats (perplexity 8.1), 52.6% right first guesses; chapter 7's network: 2.389, 47.3% |
| Test, first 100 stories (21,730 tokens) | 52.1%; chapter 7's network 46.6%; chapter 6's network and counting 41.8% |
| Test, all 1,000 stories (211,846 tokens) | 52.6%; chapter 7's network 47.3%, chapter 6's 42.2%, counting 42.1% |

Everything below runs on the page's own code: the GPT in `<script id="gpt">` of [`gpt/index.html`](../gpt/index.html), with chapter 6's tokenizer (the page carries an identical copy of `<script id="tok">`). The build scripts read that code straight out of the page.

## The data

The same files and splits as [chapter 6](chapter-6-training.md#the-data) and [chapter 7](chapter-7-training.md#the-data): training stories 2 to 20,001 of `TinyStoriesV2-GPT4-train.txt`, the first 1,000 complete stories of the validation file as test stories (the page measures on the first 100), and validation stories 1,001 onward for choosing settings. Each story is read on its own, starting with the `<end>` token, as in chapter 7. Chapter 8 ships no data of its own; the page loads `words/stories.js`.

**Made-up names.** TinyStories has few names. Of the 20,000 training stories, 13,146 (two thirds) have as their hero one of 25 one-token names (the page's list: "tim", "lily", "tom", "sue", "max", …). A name is mentioned where its token stands as a word of its own, or before an apostrophe ending ("ben ##'s"), or as its possessive when the vocabulary has that as a single token ("tim's", "lily's", "tom's", "sue's", "max's", "mia's", "sam's"), but not where its token begins a longer word, as "tom" does "tomato" or "tim" does "times"; a story's hero is the listed name it mentions most often. A network trained on these stories seldom has to copy a name it does not know, and the GPT trained on them as they are barely learned to (see the comparison run below). So in the finished run, every training story drawn for a step also draws a roll and a seed from the run's random stream; if the roll is under 0.5 and the story has a listed hero, every mention of the hero is replaced by a new made-up name (a single-token possessive by the name's pieces and "##'s", as the tokenizer cuts "curin's"; `tools/gpt/rename.js`):

- 2 or 3 syllables, each a consonant (b c d f g h j k l m n p r s t v w z) and a vowel (a e i o u), then with chance 5/8 a last consonant (n, r, l, s or x), drawn with the seed;
- cut into pieces by the tokenizer, and redrawn if it is a single token, if it is one of the 6 invented names the page tests with ("zarko", "vorna", "quilla", "morvin", "dashu", "tilda"), or if it contains any pair of their pieces (such as "z ##ark" or "##ill ##a").

Examples: "nuto", "gakobe", "dajo" ("a little boy named gakobe went to the park"). A story is drawn many times over the run, with a new roll and a new name each time; about a third of the stories in a step are renamed. The validation and test stories are never renamed; the page's test of invented names replaces a story's name by the same rule. Because the 6 test names' piece pairs were never generated, a pair the page calls "never seen in training" was seen neither in the stories nor in the made-up names.

## The network

| Part | Shape | Numbers | Starting values |
| --- | --- | --- | --- |
| Embedding table `E` | 2,000 rows × 64 | 128,000 | 0.1 × a standard normal |
| Per block (× 4): layer norm 1, gain `g1` and shift `s1` | 64 + 64 | 128 | 1 and 0 |
| Queries `Wq`, keys `Wk`, values `Wv` | 3 × 64 × 64 (4 heads of 16) | 12,288 | a standard normal ÷ √64 |
| Distance scores `R` | 4 heads × 256 | 1,024 | 0 |
| Output of attention `Wo` | 64 × 64 | 4,096 | a standard normal ÷ √64 ÷ √8 |
| Layer norm 2, `g2` and `s2` | 64 + 64 | 128 | 1 and 0 |
| MLP in, `W1` and `b1` | 256 × 64 + 256 | 16,640 | a standard normal ÷ √64; 0 |
| MLP out, `W2` and `b2` | 64 × 256 + 64 | 16,448 | a standard normal ÷ √256 ÷ √8; 0 |
| Last layer norm, `gf` and `sf` | 64 + 64 | 128 | 1 and 0 |
| Output layer, `Wout` and `bout` | 2,000 × 64 + 2,000 | 130,000 | a standard normal ÷ √64; 0 |
| Total | | 461,136 | |

The 4 blocks hold 4 × 50,752 = 203,008 numbers. Normal draws use the Box–Muller formula on the seeded stream, filling `E`, `Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`, `Wout` in that order, each kind for all 4 blocks at once. `Wo` and `W2`, whose results are added to the rows, start smaller by 1/√(2 × 4) = 1/√8, as in GPT-2.

**Close to GPT-2, with chapter 7's positions.** The blocks are GPT-2's, with the layer norm before each part ("pre-LN"), GELU in GPT-2's tanh form, and a last layer norm after the stack. Three differences: positions are marked by chapter 7's learned score for each distance back, per head, added to the scores (GPT-2 adds a learned row per absolute position to each token's row); the query, key, value and `Wo` tables have no biases (the MLP's do); and the output layer has a table of its own (GPT-2 reuses the embedding table there; chapter 6 found that sharing learned worse for its network).

## The forward pass

For the tokens t₀ = `<end>`, t₁, … of a story, the rows start as x⁰ᵢ = E[tᵢ]. Block l (l = 1 … 4) turns the rows xᵢ into new rows:

```math
\hat x_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + 10^{-5}}}, \qquad \mathrm{LN}(x_i) = g \odot \hat x_i + s \qquad (\mu_i, \sigma_i^2: \text{the mean and variance of the 64 numbers of } x_i)
```

```math
q_i, k_i, v_i = W_q\,\mathrm{LN}_1(x_i),\ W_k\,\mathrm{LN}_1(x_i),\ W_v\,\mathrm{LN}_1(x_i), \qquad \text{head } h \text{ uses numbers } 16h \dots 16h+15
```

```math
w^{(h)}_{ij} = \operatorname{softmax}_{j}\Big(\frac{q^{(h)}_i \cdot k^{(h)}_j}{\sqrt{16}} + R^{(h)}_{i-j}\Big), \qquad o^{(h)}_i = \sum_j w^{(h)}_{ij}\, v^{(h)}_j, \qquad \max(0, i-255) \le j \le i
```

```math
a_i = x_i + W_o\,[o^{(1)}_i; o^{(2)}_i; o^{(3)}_i; o^{(4)}_i], \qquad x_i \leftarrow a_i + W_2\,\mathrm{GELU}\big(W_1\,\mathrm{LN}_2(a_i) + b_1\big) + b_2
```

```math
\mathrm{GELU}(z) = \tfrac12 z \Big(1 + \tanh\big(\sqrt{2/\pi}\,(z + 0.044715\,z^3)\big)\Big), \qquad p_i = \operatorname{softmax}\big(W_{out}\,\mathrm{LN}_f(x^4_i) + b_{out}\big)
```

The loss at position i is −ln pᵢ(tᵢ₊₁) (the `<end>` closing the story, at the last position); a step's loss is the mean over all positions of its stories. `forward(m, tokens, from)` runs the blocks over every position and the last layer norm and output layer only from position `from` on, which is all the page needs to guess the next token.

## The backward pass

From the scores down: p − y at the output, through the output layer and the last layer norm, then through the blocks from the top. In each block, the gradient of the row that comes out, g, splits at each addition:

```math
\frac{\partial L}{\partial a_i} = g_i + \mathrm{LN}_2'^{\top}\, W_1^\top \big(\mathrm{GELU}'(z_i) \odot W_2^\top g_i\big), \qquad \frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial a_i} + \mathrm{LN}_1'^{\top}\big(W_q^\top \delta q_i + W_k^\top \delta k_i + W_v^\top \delta v_i\big)
```

where δq, δk and δv come from chapter 7's attention backward pass, head by head, applied to W_oᵀ ∂L/∂aᵢ. The layer norm's backward pass, for a row with x̂ and spread σ and ĝ = g ⊙ ∂L/∂LN:

```math
\frac{\partial L}{\partial x} = \frac{1}{\sigma}\Big(\hat g - \operatorname{mean}(\hat g) - \hat x\,\operatorname{mean}(\hat g \odot \hat x)\Big)
```

At the bottom, each position's gradient goes to its token's row of `E`.

## Adam and the schedule

Chapters 6 and 7's Adam (β₁ 0.9, β₂ 0.999, ε 10⁻⁸), applied to the step's gradient averaged over its positions. The rate stays at 0.004 for the first 6,650 steps, then falls linearly to 0.004 ÷ 2,850 at the last step. No weight decay, gradient clipping, warm-up or dropout.

## Training on several processor cores

`tools/gpt/ptrain.js` draws a step's stories exactly as the page's trainer does (with their rolls and seeds for made-up names), and sorts them longest first. Each worker thread takes the next story not yet taken, renames its hero if its roll says so, and works out that story's gradient into a slot of its own (one slot per story of the step). Then each worker adds up the slots, in the batch's order, for its share of the weights, and applies the Adam update to it. These are the same additions in the same order as the page's `trainStep`, which also works out each story's gradient on its own and adds them in turn, so the result is bit-identical to it for any number of workers (`train.js cmp=…`: identical after 6 steps with 16 workers, with 3, and with made-up names).

A story's forward and backward passes cannot be shared between workers without repeating the lower blocks for every share, so whole stories are the unit of work. With 16 stories per step and 16 workers, a step waits for its longest story: the run without made-up names, alone on the machine, made about 6,950 positions a second (one thread makes about 1,600), against about 14,400–15,100 with 64 stories per step. 64 stories per step learned much worse per position (next section), so the run uses 16.

## How the settings were chosen

The shape was chosen up front: GPT-2's block, at a size close to chapters 6 and 7's networks (4 blocks, rows of 64, 4 heads of 16, an MLP of 256, 4 times the row as in GPT-2). The batch and the learning rate were chosen with short runs of about 4.1 million positions each (the same schedule shape as the final run, falling over the last 30%), measured on all tokens of validation stories 1,001 to 1,300, without made-up names:

| Question | Compared | Validation accuracy after about 4.1M positions | Kept |
| --- | --- | --- | --- |
| Stories per step, at 0.004 | 64 (250 steps), 32 (500 steps), 16 (1,000 steps) | 33.3, 38.9, 42.4% | 16 |
| Learning rate, 64 stories per step | 0.004, 0.008, 0.016 | 33.3, 32.5, 28.9% | |
| Learning rate, 16 stories per step | 0.002, 0.004, 0.008 | 39.6, 42.4, 41.6% | 0.004 |

At this size the network learns far more per position from many small steps than from few large ones, and higher rates do not make up for it. After the same 4.1 million positions, chapter 7's network was at about 43% on the same validation stories (16 stories per step is twice its 8).

## The final run and its results

The validation accuracy over the run (stories 1,001–1,300): 19.9% after 50 steps, 40.9% after 1,000, 47.4% after 3,000, 52.6% at the end.

**The ladder.** On the page's 21,730 test tokens (every token of the first 100 test stories), by where the right answer last appeared (`compare.js`):

| Where the right answer last appeared | Share | Counting, last 4 | Chapter 6 | Chapter 7 | GPT | GPT without made-up names |
| --- | --- | --- | --- | --- | --- | --- |
| All tokens | 100% | 41.8% | 41.8% | 46.6% | 52.1% | 51.9% |
| Among the last 8 tokens | 9.6% | 55.1% | 55.6% | 60.1% | 67.2% | 65.8% |
| Earlier in the story, further back | 46.2% | 42.8% | 44.2% | 52.6% | 59.5% | 59.3% |
| … and not one of the 100 most common tokens | 11.2% | 21.1% | 20.0% | 36.8% | 46.3% | 43.2% |
| … and the story's name | 1.6% | 3.5% | 2.6% | 45.3% | 55.0% | 62.1% |
| Nowhere earlier in the story | 44.2% | 37.8% | 36.4% | 37.4% | 41.2% | 41.2% |

Over all 1,000 test stories: 52.6% (without made-up names 52.4%), chapter 7 47.3%, chapter 6 42.2%, counting 42.1%; for the story's name, 52.7% (56.8%), against 46.4% for chapter 7. The made-up names cost the GPT some accuracy on common names (55.0% against 62.1% on the page's stories, 52.7% against 56.8% on all 1,000) and bought it the copying of new ones. The story's name is marked as chapter 7 marks it (the listed one-token name whose token comes up most often, wherever it stands), so that the two chapters' numbers match; among the page's 340 name tokens, this counts one "tom" that begins "tomato".

**Invented names** (`induction.js`: each of the 6 invented names given to each of the 63 name stories, at every mention of the hero as a word, by the rule above; 3,216 pieces whose pair with the piece before is never seen in training, at later mentions, and 1,608 whose pair is):

| Network | Never-seen pairs, first mention | Never-seen pairs, later mentions | Seen pairs, first | Seen pairs, later | Repeated random tokens, second copy |
| --- | --- | --- | --- | --- | --- |
| Chapter 7 | 0.0% | 0.0% | 23.8% | 40.0% | 0.3% |
| Chapter 7, retrained with made-up names | 0.4% | 16.2% | 15.9% | 46.3% | 0.3% |
| GPT without made-up names | 0.0% | 4.0% | 22.6% | 46.1% | 0.2% |
| GPT | 1.0% | 61.1% | 7.9% | 79.7% | 1.2% |

The page's own measure gives each name story one of the 6 names (542 never-seen pieces at later mentions): GPT 61.4%, chapter 7's network 0.0%. Chapter 7's network retrained with the same made-up names (`F8R`, 19,000 steps of 8 stories at 0.008, validation 46.3%) gets 14.8% right, but in another 30.1% its first guess is a piece of the same name in the wrong place: it learned to bring back pieces of a name from the story, by what they are, but not which comes next. The GPT puts a wrong piece of the name first 0.2% of the time.

At a first mention, the GPT gets 1.0% of the never-seen pairs (1.2% by the page's measure), where the GPT without made-up names gets 0.0%. Nothing in the story can help there: it learned what made-up names tend to look like, and after a piece that ends in a consonant it usually guesses a vowel ("##i" after "z" in 57 of the 63 stories, "##u" after "t" in 58). The few it gets right are "##u" after "da ##sh", in 5 of the 63 stories. Seen pairs at a first mention fell from 22.6% to 7.9%, since a hero's name could now be anything.

Per name, at later mentions, the GPT finishes: "vorna" 94.9% of the pieces, "morvin" 82.7%, "dashu" 67.8%, "quilla" 59.8%, "tilda" 49.6%, "zarko" 49.0%. In the last two, one piece is always missed (0 of 402 later mentions): after "z", "##ark" (0.0%; its first guess is "##ele" in 225 of the 402 later mentions), and after "t", "##ild" (0.0%; "##ot" in 208). In "dashu" and "quilla" the middle piece comes back less often ("##sh" 44.3%, "##ill" 20.6%). The copy is read: at all 402 later mentions of "zarko", block 2's heads 4 and 1, reading "z", put on average 95% and 80% of their weight on the earlier "##ark"s (at "tilda", 90% and 92% on the earlier "##ild"s), yet the GPT gives "##ark" 0.45% on average and "##ild" 0.12%. Neither wrong guess ever follows its first piece in the training stories, and in 200,000 made-up names drawn by the same rule, "##ele" comes after a first piece "z" in 0.2% of the names that start with "z", and "##ot" never after a first piece "t"; this write-up does not explain why these pieces lose. The page's film and opening example use "vorna", the name it finishes best, both of whose pairs are never seen in training; the page shows every name's rate.

On repeated random tokens (40 common tokens, then the same 40), the GPT copies little: 1.2% of the second copy. Its copying is for names, which the training stories asked for; a list of random words, unlike anything in them, does not set it off.

**The heads** (`heads.js`). At later mentions of the invented names (802 positions), each head's average weight on the token just before and on the token right after the earlier copy of the token being read:

| | Head 1 | Head 2 | Head 3 | Head 4 |
| --- | --- | --- | --- | --- |
| Block 1 | 3%, 1% | 37%, 1% | 50%, 0% | 37%, 2% |
| Block 2 | 0%, 26% | 1%, 3% | 16%, 2% | 0%, 29% |
| Block 3 | 0%, 15% | 3%, 2% | 1%, 5% | 27%, 0% |
| Block 4 | 37%, 0% | 22%, 1% | 2%, 1% | 8%, 1% |

The page tints a head as a previous-token head at 30% or more on the token before, and as an induction head (block 2 or above) at 20% or more on the token after the earlier copy: block 1's heads 2, 3 and 4 and block 4's head 1 are previous-token heads, block 2's heads 1 and 4 induction heads. On repeated random tokens, block 2's heads 1 and 4 put only 5% and 11% on the token after the earlier copy. The GPT without made-up names has similar previous-token heads (block 1, heads 2 to 4: 27–36%), and two heads that lean a little toward the earlier copy (block 3 head 3, 19%; block 2 head 4, 12%), but it finishes only 5.4% of the never-seen pairs.

**Switching heads off** (`ablate.js`; a head's value table set to 0), with the page's measure of invented names (never-seen pairs, later mentions) and every test token:

| Head off | Names | All test tokens |
| --- | --- | --- |
| none | 61.4% | 52.1% |
| block 1, head 3 (previous token) | 0.0% | 40.2% |
| block 2, head 4 (induction) | 4.6% | 50.5% |
| block 2, head 1 (induction) | 49.8% | 51.5% |
| block 1, head 2 (previous token) | 65.5% | 49.7% |
| block 1, head 4 (previous token) | 54.8% | 47.1% |
| block 4, head 1 (previous token) | 63.1% | 50.2% |
| any other one head | 58.1–62.5% | 49.3–51.4% |

Block 2, head 4 is the network's induction head in the sense that matters: without it, names drop to 4.6% while every token together drops only 1.6 points. Block 2, head 1 reads the earlier copy almost as much (26% against 29%), but the network relies on it far less: without it, names fall to 49.8%. Block 1, head 3 is the previous-token head the pair needs: without it, names stop altogether, and it matters for much else too (all tokens 40.2%). The other previous-token heads are not needed in the same way: without block 1's head 4, names fall to 54.8%; without block 1's head 2 or block 4's head 1, they rise a little.

**When it appeared.** The page's measure at every checkpoint (`package.js`; the same at every 4th checkpoint again in `verify.js`): never-seen pairs at later mentions 0.0% at step 250, 1.1% at 500, 9.4% at 750, 0.0% at 1,000, 14.0% at 1,250, 30.1% at 2,000, 41.9% at 3,000, 46.5% at 3,750, 53.0% at 4,000, then between 52.4% and 66.2% to the end (66.2% at step 8,750, 61.4% at step 9,500). The page takes the mean of the last four checkpoints, 62.5%, as the final rate: a fifth of it was first reached at step 1,250 and four fifths at step 4,000. The second copy of repeated tokens stayed between 0.0% and 1.3% from step 250 on. Without made-up names, never more than 10.9% and 0.4%. So the copying came gradually and unevenly, over about 2,750 steps, not in a sudden jump. In the live experiment ("Training in the reader's browser", below), the second copy goes from 20% to 80% right in 30 to 50 steps (`toy.js`, 8 seeds).

**The logit lens** (`lens.js`): the first guess read after each block, over the page's 21,730 test tokens: the token's own row 0.7%, after block 1 22.8%, block 2 32.2%, block 3 40.5%, block 4 52.1%. For "once upon a", the top guess after each block is "##qu", "castle", "village", "time", "time".

**Without residual connections, without layer norms.** The same network, from the same seed and with the same stories and made-up names, for 3,000 steps at a constant 0.004 (the finished run's rate is also constant until step 6,650):

| Step | GPT | Without residual connections | … and with ordinary starting sizes | Without layer norms |
| --- | --- | --- | --- | --- |
| 1,000 | 40.9% | 7.5% | 7.5% | 40.3% |
| 2,000 | 45.3% | 7.5% | 7.5% | 45.3% |
| 3,000 | 47.4% | 7.5% | 7.5% | 47.6% |

Without residual connections, the network was stuck from the first checkpoint (step 250) at guessing the most common token, with a validation loss of about 5.73 nats, and it stayed stuck when `Wo` and `W2` started at the ordinary 1/√64 and 1/√256 instead of GPT-2's 1/√8 smaller (`variant=nores1`). The size of the gradient reaching each block's weights, summed over 16 training stories at step 3,000 (`grads.js`):

| | Block 1 | Block 2 | Block 3 | Block 4 |
| --- | --- | --- | --- | --- |
| GPT | 1,168 | 467.3 | 427.7 | 446.1 |
| Without residual connections | 0.000111 | 0.00296 | 0.470 | 76.5 |
| Without layer norms | 941.5 | 838.4 | 864.4 | 882.9 |

Without residual connections, block 1 gets about 1/690,000 of block 4's gradient: chapter 4's vanishing gradients, in a transformer. Without layer norms, this small network trains as well over these 3,000 steps (47.6% against 47.4%).

## From checkpoint to page

`tools/gpt/package.js` writes `gpt/weights.js` (2,467,728 bytes): the GPT's 461,136 numbers rounded to 32-bit floats (every number the page shows is measured with these), the run's validation log, the page's measure at every checkpoint of the run and of the run without made-up names (44 each), the logs of the three comparison runs with the gradient sizes above, and the retrained chapter 7 network's rates. The page loads it with chapter 6's stories and network (`words/stories.js`, `words/weights.js`) and chapter 7's network (`attention/weights.js`), which it runs with a compact copy of chapter 7's forward pass.

## Training in the reader's browser

The page's "Train" button runs the same code in a Web Worker: a fresh GPT from seed 2 (each Reset moves to the next seed), 8 whole stories per step picked as above, with made-up names given the same way (the page carries the same rules for names as `rename.js`, and its worker uses them; `check7.js` checks that the page renames every training story exactly as `rename.js` does), at a constant learning rate of 0.004. Every 60 seconds the page measures it on the first 20 test stories.

The live experiment of Secret 3 is the same code with other settings: a GPT of 2 blocks, 2 heads each, rows of 32 and no MLP neurons (`M = 0`: each block's MLP is reduced to its bias `b2`, a learned row added at every position), a vocabulary of 32 (`<end>` and 31 words), and a window of 49. Each training sequence is `<end>`, then 8 to 24 different words drawn at random, then the same words again; 16 sequences per step, Adam at 0.01, from seed 1 (each Reset moves to the next seed). In the second copy, from its second word on, neither knowing the words nor counting places can help: only finding the earlier copy can. The second copy's first word cannot be told from anything before it, so the page and `toy.js` measure the second copy from its second word. Tested in Node with 8 seeds (`toy.js`, the same code and data): the second copy, from its second word, passed 50% right between steps 550 and 1,010 every time, out of a plateau at about 1 in 31, and ended at 100%. In every seed, both heads of block 1 became previous-token heads (25–48% of their weight on the token before) and block 2 had an induction head (94–100% on the token after the earlier copy); in 6 of the 8, both of block 2's heads did. With the repeat at one fixed length, a head learned to read 15 tokens back instead (a shortcut that is not induction), which is why the length varies. In headless Chrome the page makes about 28 steps a second; with seed 1 the jump came at about step 970, and both heads' weights rose with it.

## Checks

| Check | How | Result |
| --- | --- | --- |
| Backward pass | `gradcheck.js`: central differences on small networks, every weight array, windows shorter than the story, outputs from a later position, no MLP, and both variants | largest relative difference 1.9 × 10⁻⁵ (h = 10⁻⁵, over gradients above 10⁻⁵, where the differences can resolve them) |
| Backward pass, in the page | at every load, 54 weights from every part of the finished GPT against central differences | largest relative difference 3 × 10⁻⁷ |
| Python translation | `pycheck.js`: the page's Python against its JavaScript on the real shapes | starting weights within 1.1 × 10⁻¹⁶, probabilities within 6 × 10⁻¹⁸, losses identical to 9 decimals, weights after 3 Adam steps within 4.4 × 10⁻¹¹ |
| Multi-core trainer | `ptrain.js` with 3 and 16 workers, with made-up names, 6 steps, against `train.js cmp=…` (the page's `trainStep`) | bit-identical (no weight differs in any bit) |
| Chapter 7 and 6 in the page | `check7.js`: the page's `forward7` against chapter 7's `forward`; `probs6` and the tokenizer against chapters 6 and 7 | identical probabilities; identical code |
| Made-up names | `check7.js`: the page's rules for names and its renaming against `rename.js`, on every training story renamed with chance one half and for sure; the page's test stories against `lib.js`'s | identical in all 40,000 renamings and all 390 test (story, name) pairs; no mention of a renamed hero left, single-token possessives included |
| Packaging | `verify.js`: `weights.js` against the checkpoint; the stored measurements against fresh ones (every 4th checkpoint and the last) | equal up to 32-bit rounding (largest difference 2.3 × 10⁻⁷); moments equal to their 4 stored decimals; the gradient sizes and the retrained chapter 7 rates identical |
| Live against offline | the page's numbers in headless Chrome against `compare.js`, `package.js`'s measure, `heads.js`, `ablate.js` and `lens.js` | the same (52.1%, 61.4%, every head's weights, every head switched off, the logit lens) |
| Reviews | an independent accuracy review of the page and a code review | findings fixed before publishing |

## How to reproduce it

The commands run from `tools/gpt/` and need Node 24 or later and chapter 6's token files (run chapter 6's `fetch.sh` and `prep.js` first, see [chapter 6's write-up](chapter-6-training.md#how-to-reproduce-it)); `pycheck.js` also needs Python 3. Checkpoints and logs go to `tools/gpt/work/`, which git ignores.

```sh
cd tools/gpt
mkdir -p work/runs
node ptrain.js lr=0.004 B=16 steps=9500 every=250 early=0,25,50,100,150,200 rename=0.5 save=R > work/runs/R.txt
                              # the finished GPT: about 95 minutes with 16 workers on an Apple M5 Pro, alone
node ptrain.js lr=0.004 B=16 steps=9500 every=250 early=0,25,50,100,150,200 save=G > work/runs/G.txt
                              # the same without made-up names (Secret 3)
node ptrain.js lr=0.004 B=16 steps=3000 decay=0 every=250 rename=0.5 variant=nores save=Rnr > work/runs/Rnr.txt
node ptrain.js lr=0.004 B=16 steps=3000 decay=0 every=250 rename=0.5 variant=nonorm save=Rnn > work/runs/Rnn.txt
node ptrain.js lr=0.004 B=16 steps=3000 decay=0 every=250 rename=0.5 variant=nores1 save=Rnr1 > work/runs/Rnr1.txt
                              # without residual connections, without layer norms, and without residual
                              # connections with ordinary starting sizes (Secret 5)
(cd ../attention && node ptrain.js lr=0.008 steps=19000 rename=0.5 workers=8 save=F8R > work/runs/F8R.txt)
                              # (chapter 7's trainer is exact for a given number of workers; the run used 8)
                              # chapter 7's network with the same made-up names (Secret 1)
node package.js               # writes gpt/weights.js
node verify.js every=4        # weights.js against the checkpoint; the stored measurements against fresh ones
node gradcheck.js             # the backward pass against central differences
node pycheck.js               # the Python translation against the JavaScript
node check7.js                # the page's copies of chapters 6 and 7's networks and of the made-up names
node compare.js R_9500.bin G_9500.bin   # counting, chapters 6 and 7 and the GPT, by where the answer last appeared
NSTORIES=1000 node compare.js R_9500.bin G_9500.bin   # the same over all 1,000 test stories
node induction.js ch7=1 R_9500.bin G_9500.bin   # invented names (all 6 in every name story) and repeated tokens
CH7=F8R_19000.bin node induction.js ch7=1   # the same for chapter 7's network retrained with made-up names
node heads.js R_9500.bin      # where each head's weight goes (and G_9500.bin for the run without made-up names)
node ablate.js R_9500.bin     # each head switched off (Secret 2)
node lens.js R_9500.bin       # the logit lens (Secret 4)
node grads.js                 # the gradient reaching each block, with and without residual connections and layer norms
node toy.js seeds=8           # the live experiment of Secret 3, in Node, for 8 seeds
```

`node train.js` trains with the page's own `trainStep` on one thread. `look.js` opens the served page in headless Chrome (serve the repo with `python3 -m http.server 8765`) and prints its live readouts; `shots_anim.js` screenshots every scene of the page's three animations.
