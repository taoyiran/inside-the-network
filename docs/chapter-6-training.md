# How Chapter 6's network is trained

A complete account of how the next-token network in [Chapter 6, Words as vectors](https://taoyiran.github.io/inside-the-network/words/) was trained, measured and shipped, as of 12 October 2026. Everything here can be rerun with the scripts in [`tools/words/`](../tools/words/); the last section gives the commands.

## Overview

Chapter 6's finished network learned to guess the next token of children's stories. It trained for 600,000 steps of 64 examples on 8 processor cores, in 54 minutes. On 100 stories it never saw, its first guess is right 41.8% of the time, the same as a careful counting method.

Everything below runs on the page's own code: the tokenizer in `<script id="tok">` and the network in `<script id="nwm">` of [`words/index.html`](../words/index.html). The build scripts in `tools/words/` read that code straight out of the page.

| Item | Value |
| --- | --- |
| Training text | 20,000 TinyStories stories, 4,300,373 tokens |
| Vocabulary | 2,000 tokens, from 1,922 byte-pair merges |
| Network | last 8 tokens → 8 rows of 32 numbers → 128 tanh neurons → 2,000 scores → softmax |
| Parameters | 354,896 |
| Loss | cross-entropy: −log of the probability given to the real next token |
| Optimizer | Adam (β₁ 0.9, β₂ 0.999, ε 10⁻⁸), learning rate 0.001, falling linearly to 0 over the last 30% of steps |
| Batch | 64 positions drawn at random from the training tokens |
| Length | 600,000 steps = 38,400,000 examples, about 8.9 passes over the training tokens |
| Randomness | one seed (1) sets the starting weights and every batch |
| Hardware and time | 8 worker threads on an Apple M5 Pro, 3,240 seconds |
| Validation, stories 1,001–2,000 | loss 2.763 nats (perplexity 15.8), 41.7% right first guesses |
| Test, first 100 stories (21,730 tokens) | 41.8% right first guesses; counting with the last 4 tokens also 41.8% |

## The data

The network learns from 20,000 short stories written by GPT-4 and is tested on 1,000 others. Validation uses a further 1,000.

- **Source.** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), by Ronen Eldan and Yuanzhi Li at Microsoft Research (2023, arXiv:2305.07759). GPT-3.5 and GPT-4 wrote the collection, using only words a 3- or 4-year-old understands. The V2 files used here hold only GPT-4 stories.
- **License.** [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/). `words/stories.js` is shared under the same agreement, with a notice of how it was changed. The trained weights count as "results", on which the agreement places no conditions (section 3.5). They are MIT, like the code.
- **Download** (`tools/words/fetch.sh`). The whole validation file, `TinyStoriesV2-GPT4-valid.txt` (22.5 MB), and the first 150 MB of `TinyStoriesV2-GPT4-train.txt`. Only the start of the training file is needed.
- **Splitting into stories.** The text is cut at every `<|endoftext|>` and each story is trimmed of blank space. The text before the first marker and after the last is dropped. In the validation file, the text before the first marker is the end of a cut-off story. In the training file it is a complete story, so story 1 is skipped.

| Set | Stories | Tokens | Used for |
| --- | --- | --- | --- |
| Training | stories 2 to 20,001 of the training file | 4,300,373 | training, the counting baselines, live training on the page |
| Test | the first 1,000 complete stories of the validation file | 211,846 | shipped in `stories.js`; the page measures on the first 100 (21,730 tokens) |
| Validation | stories 1,001 to 2,000 of the validation file | 212,511 | choosing among trained networks, and the learning curve |

A training story averages 215 tokens. The training stories use 9,597 different words, counting forms such as "lily's" separately. The 1,000 test stories bring 99 words the training stories never use, such as "tulip", "pudding" and "excellent".

## The tokenizer

Text becomes token numbers in three steps: tidy it, split it into words and marks, and cut each word into learned pieces. The vocabulary has 2,000 tokens, and 92.9% of the words in the test stories are a single token.

### 1. Tidy (`tidy`)

1. Decompose accented letters (Unicode NFKD) and drop the accent marks (U+0300 to U+036F), so "café" becomes "cafe".
2. Lowercase everything.
3. Turn curly single quotes, the backtick and the acute accent into `'`, curly double quotes into `"`, and … into three dots.
4. Turn an en dash, an em dash or a run of two or more hyphens, with the spaces around it, into " — ". A hyphen with a space on each side also becomes " — ".
5. Replace every character other than a–z, 0–9, `. , ! ? ; : ' " -`, the em dash, line breaks and spaces with a space.
6. Collapse runs of spaces and tabs to one space, collapse blank lines and the spaces around line breaks to a single line break, and trim the ends.
7. Give quotation marks a direction, line by line. A `"` becomes “ when the character written just before it is a space, an em dash, “ or ‘, or when it starts the line; otherwise it becomes ”. A `'` becomes ‘ when it follows one of those and comes before a letter. The next `'` not followed by a letter then becomes ’. Every other `'` stays an apostrophe.

### 2. Split into words and marks (`pieces`)

The regular expression `[a-z]+(?:'[a-z]+)*|[^ a-z]` finds the pieces. A word is a run of letters, with apostrophes allowed inside ("don't", "lily's"). Every digit, punctuation mark and line break is a piece of its own. Spaces are dropped, because simple rules put them back.

### 3. Learn the pieces: byte-pair encoding (`learnMerges`)

1. Count how often each word appears in the training stories: 9,597 different words.
2. Start every word as its letters. The first letter stands alone and each later one carries "##", meaning "glued to the piece before": "play" is `p ##l ##a ##y`.
3. Count every pair of neighboring pieces inside every word, weighting each word by how often it appears.
4. Glue the most frequent pair into one new piece everywhere it occurs. On a tie, the pair whose text sorts first wins.
5. Repeat steps 3 and 4, 1,922 times. Merges never cross from one word to the next, and marks are never merged.

The first merges are `##h + ##e → ##he`, then `t + ##he → the`, then `##n + ##d → ##nd`. Learning all 1,922 takes about 3 seconds.

### 4. The vocabulary (`makeTokenizer`)

A token's number is its place in this list, and that number is all the network gets.

| Numbers | Tokens |
| --- | --- |
| 0 | `<end>`, the end of a story |
| 1 | the line break |
| 2–14 | 13 marks: `. , ! ? ; : '` “ ” ‘ ’ `-` — |
| 15–24 | the digits 0 to 9 |
| 25–50 | the 26 letters that start a word |
| 51–77 | the 27 glued-on pieces `##a` to `##z` and `##'` |
| 78–1,999 | the 1,922 merged pieces, in the order they were learned |

### 5. Cutting text into tokens, and back (`encode`, `decode`)

- **A mark** becomes its number directly.
- **A word** is cut by `cutWord`: start from its letters, then keep applying the earliest-learned merge among neighboring pairs until none applies. Each word's result is cached. "towers" becomes `tow ##ers`, and "discovered" becomes `dis ##co ##ver ##ed`.
- **Every story** is followed by token 0 in the data files.
- **Decoding** glues the "##" pieces on and puts a space between tokens, except before `. , ! ? ; : '` ” ’ `-` and a line break, after “ ‘ `-` and a line break, and between digits. `<end>` becomes a blank line.
- **The round trip** gives back the tidied text exactly for 20,931 of the 21,000 stories. The other 69 come back slightly different, from spellings such as a space just inside a quotation mark or dots run into the next word. Typed text can differ too: "3.50" comes back as "3. 50".

GPT's tokenizers work the same way on bytes instead of letters. They keep capitals and spaces inside their tokens, and have about 50,000 to 200,000 tokens.

## Training examples

Every position in the 4,300,373 training tokens is one example: the 8 tokens before it are the input, and the token at it is the answer.

- **The context.** The 8 tokens before the position, oldest first. Places before the story's start are filled with token 0 (`<end>`), so the first token of a story has a context of eight `<end>` tokens. This is the page's `contextAt` (and `ctxAt` in the tools, which gives the same contexts).
- **The answer** can be any token, including the `<end>` that closes a story, so the network also learns where stories stop.
- **The batch.** 64 positions drawn uniformly at random, with replacement, as `floor(r() × 4,300,373)`. `r` is the mulberry32 generator with seed 1.
- **One random stream.** The same generator first draws the starting weights, then 64 positions per step. There are no epochs and no shuffled passes: over 600,000 steps a position is drawn 8.9 times on average, some more often and some never.

## The network

The network is an embedding table, one hidden layer of 128 tanh neurons, and an output layer that scores all 2,000 tokens: 354,896 numbers to learn.

| Part | Shape | Numbers | Starting values |
| --- | --- | --- | --- |
| Embedding table `E` | 2,000 rows × 32 | 64,000 | 0.1 × a standard normal |
| Hidden weights `W1` | 128 × 256 | 32,768 | a standard normal ÷ √256 (spread 0.0625) |
| Hidden biases `b1` | 128 | 128 | 0 |
| Output weights `W2` | 2,000 × 128 | 256,000 | a standard normal ÷ √128 (spread 0.088) |
| Output biases `b2` | 2,000 | 2,000 | 0 |
| Total | | 354,896 | |

- **Normal draws** use the Box–Muller formula on the seeded stream, `√(−2 ln(1 − u₁)) × cos(2π u₂)`, filling `E`, then `W1`, then `W2`. The biases take no draws.
- **The table is separate from the output layer.** Reusing the table to score tokens, as GPT-2 does, learned worse here (see "How the settings were chosen").
- **Cost.** A forward pass is about 289,000 multiply-adds (256 × 128 + 128 × 2,000), and the backward pass about twice that.

## The forward pass and the loss

For a context of tokens t₁ (oldest) to t₈ (newest), the network lays the 8 rows side by side, passes them through the hidden layer, scores every token and turns the scores into probabilities.

```math
x = \big[\,E_{t_1};\ E_{t_2};\ \dots;\ E_{t_8}\,\big] \in \mathbb{R}^{256}
```

```math
a_u = \tanh\Big(b^{(1)}_u + \sum_{i=1}^{256} W^{(1)}_{u,i}\, x_i\Big), \qquad u = 1, \dots, 128
```

```math
s_v = b^{(2)}_v + \sum_{u=1}^{128} W^{(2)}_{v,u}\, a_u, \qquad v = 1, \dots, 2000
```

```math
p_v = \frac{e^{\,s_v - \max_w s_w}}{\sum_{w} e^{\,s_w - \max_w s_w}}, \qquad L = -\ln p_{\text{next}}
```

Subtracting the largest score before exponentiating keeps the exponentials from overflowing and does not change the probabilities. The loss is in nats, and a batch's loss is the mean over its 64 examples. Perplexity is e raised to the loss: the final 2.763 nats is a perplexity of 15.8, as unsure as a fair choice among 16 tokens.

## The backward pass

The gradient starts at the scores and runs back through every layer by the chain rule, ending in the 8 rows that were read.

```math
\frac{\partial L}{\partial s_v} = p_v - [\,v = \text{next}\,]
```

```math
\frac{\partial L}{\partial b^{(2)}_v} = \frac{\partial L}{\partial s_v}, \qquad \frac{\partial L}{\partial W^{(2)}_{v,u}} = \frac{\partial L}{\partial s_v}\, a_u, \qquad \frac{\partial L}{\partial a_u} = \sum_{v} W^{(2)}_{v,u} \frac{\partial L}{\partial s_v}
```

```math
\frac{\partial L}{\partial z_u} = \frac{\partial L}{\partial a_u}\,\big(1 - a_u^2\big), \qquad \frac{\partial L}{\partial b^{(1)}_u} = \frac{\partial L}{\partial z_u}, \qquad \frac{\partial L}{\partial W^{(1)}_{u,i}} = \frac{\partial L}{\partial z_u}\, x_i
```

```math
\frac{\partial L}{\partial x_i} = \sum_{u} W^{(1)}_{u,i} \frac{\partial L}{\partial z_u}, \qquad \frac{\partial L}{\partial E_{t_j}} \mathrel{+}= \Big(\frac{\partial L}{\partial x}\Big)_{32(j-1)+1\,\dots\,32j}
```

- **Every output weight and bias** gets a gradient from every example, because all 2,000 scores enter the softmax.
- **Only the 8 rows that were read** get a gradient for the table. A token that appears twice in the context gets both pieces added to its row.
- **A row learns from what follows it.** A token's row changes only to help guess the tokens that come after it within the next 8 positions. So words followed by similar tokens get pushed in similar directions.
- The gradients of the 64 examples are added up, then divided by 64 in the update.

## Adam and the learning-rate schedule

Every weight takes its own step size: Adam divides the step by the typical size of that weight's recent gradients. For a weight with batch-mean gradient g at step t (counted from 1):

```math
m \leftarrow 0.9\, m + 0.1\, g, \qquad s \leftarrow 0.999\, s + 0.001\, g^2
```

```math
w \leftarrow w - \eta_t \cdot \frac{m / (1 - 0.9^{t})}{\sqrt{s / (1 - 0.999^{t})} + 10^{-8}}
```

```math
\eta_t = \begin{cases} 0.001 & t \le 420{,}000 \\[2pt] 0.001 \cdot \dfrac{600{,}000 - t + 1}{180{,}000} & t > 420{,}000 \end{cases}
```

- **The averages** m and s start at 0 for every weight. The divisions by (1 − 0.9ᵗ) and (1 − 0.999ᵗ) make up for that start in the early steps.
- **Every weight is updated every step.** A table row that was not read has g = 0, but its m still carries recent gradients and shrinks by 0.9 a step, so the row keeps drifting for a few steps after it was last read.
- **Why Adam.** The row of "the" gets a gradient in almost every batch. The row of "danger", seen 55 times in 4.3 million tokens, gets one about once every 150 batches. Plain gradient descent gives both the same step size per unit of gradient; Adam lets both move at a useful pace. It also learned much faster here (see "How the settings were chosen").
- **The schedule.** The rate stays at 0.001 for the first 70% of the steps, then falls linearly to 0.001 ÷ 180,000 at the last step.
- **Not used:** weight decay, gradient clipping, dropout.

## Training on several processor cores

The finished network was trained by `tools/words/ptrain.js`. It shares out each step among 8 worker threads and gives the same result as the page's one-thread `trainStep`, except for the order of a few additions.

1. **Shared memory.** The weights, Adam's m and s, and one gradient buffer per worker live in `SharedArrayBuffer`s that every thread sees.
2. **The batch.** The main thread draws the 64 positions exactly as the page's trainer does (same seed, same order) and writes their contexts and answers to shared memory.
3. **Gradients.** Worker i takes examples i, i + 8, i + 16 and so on, 8 in all. It clears its gradient buffer and runs the page's `forward` and `backward` into it.
4. **The update.** Worker i takes one eighth of the 354,896 weights. For each one, it adds up the 8 workers' gradients in worker order, divides by 64, and applies the page's Adam update.
5. **The next step** starts once every worker has reported back, through `Atomics.wait` and `Atomics.notify`.

With 1 worker the result is bit-identical to the page's `trainStep` (checked after 60 steps). With 8 workers the largest weight difference after 60 steps was 3.5 × 10⁻¹⁴, from adding the gradients in a different order. The final run made about 185 steps a second (11,800 examples a second), against about 2,400 examples a second on one thread.

Every 10,000 steps, and at steps 1,000 and 3,000, the trainer measures the network on the validation stories and saves a checkpoint: a 64-byte JSON header with the shape, then `E`, `W1`, `b1`, `W2` and `b2` as 64-bit numbers.

## Measuring during training

During training the network is measured on 20,000 positions evenly spaced through the validation stories (stories 1,001 to 2,000 of the validation file). Two numbers are recorded: the mean loss, and the share of positions where the most probable token is the real next one.

| Steps | Examples | Validation loss (nats) | Right first guesses |
| --- | --- | --- | --- |
| 1,000 | 64,000 | 4.464 | 23.3% |
| 3,000 | 192,000 | 3.882 | 27.8% |
| 10,000 | 640,000 | 3.406 | 33.3% |
| 40,000 | 2,560,000 | 3.076 | 37.1% |
| 150,000 | 9,600,000 | 2.943 | 39.3% |
| 300,000 | 19,200,000 | 2.905 | 39.6% |
| 420,000 | 26,880,000 | 2.902 | 39.8% |
| 500,000 | 32,000,000 | 2.840 | 40.6% |
| 600,000 | 38,400,000 | 2.763 | 41.7% |

At the constant rate of 0.001, accuracy barely moves after step 150,000: from 39.3% to 39.8% at step 420,000. The fall of the learning rate over the last 180,000 steps adds 1.9 points.

## How the settings were chosen

Each choice was settled by training the candidates side by side and comparing them at the same number of examples. Rows marked * were measured on 20,000 positions spread over all 2,000 validation-file stories, which include the page's 1,000 test stories. The rest were measured on stories 1,001 to 2,000 only. Differences of a few tenths of a point are within noise.

| Question | What was compared | Result | Kept |
| --- | --- | --- | --- |
| Plain gradient descent: rate and batch * | batches of 16 at rates 0.3, 0.5, 1, 2, 4, 8; batches of 1 and 4 | 0.3 best (28.3% after 320,000 examples); 2 and above blew up; batches of 1 and 4 got worse as they trained | batches of 64 |
| Plain gradient descent, batch 64 * | rates 1 and 2 | rate 1: 32.0% after 1,280,000; rate 2 unstable (26.8%) | rate 1, as the baseline |
| Starting scale of the table * | 1, 0.1, 0.01 | 1 learned far more slowly (loss 5.05 after 64,000 examples); 0.1 and 0.01 alike (26.5% and 26.3% after 320,000) | 0.1 |
| Context length, plain gradient descent * | 4, 6, 8 tokens | 32.0%, 32.3%, 32.1% after 1,280,000 | 8 |
| Optimizer * | Adam at 0.002 against plain gradient descent at 1 | Adam 35.2% after 960,000; plain gradient descent 32.1% after 1,280,000 | Adam |
| Adam's rate * | 0.004, 0.002, 0.001 | 0.004: 33.2% after 960,000; 0.002: 36.5% and 0.001: 38.2% after 3,840,000 | 0.001 |
| Table reused as the output layer (GPT-2's tying) * | tied against separate, with Adam | tied 32.0%, separate 35.0% after 1,120,000; tied with plain gradient descent unstable | separate |
| Hidden neurons * | 128 against 256 | 38.2% and 38.0% after 3,840,000; 512 makes a guess take about 1 ms in a browser | 128 |
| Dropout | 0.3 on 256 neurons *; 0.2 on 128 neurons | 36.2% against 37.6% after 2,560,000; 38.1% against 39.3% after 9,600,000 | none |
| Context of 16 tokens | 16 against 8 | 38.3% against 37.9% after 3,840,000; 39.6% against 39.1% after 11,520,000 | 8: under half a point, for a bigger page |
| Length * | runs planned for 300,000 steps | stopped early, still improving at 6,400,000 examples (39.0%) | 600,000 steps, which ended at 41.7% |

## The final run and its results

The finished network ties counting overall, loses to it where a phrase was seen in training, and beats it on phrases the training stories never contain.

On the page's 21,730 test tokens (every token of the first 100 test stories), "right first guess" means the most probable token is the real next one:

| Method | Right first guesses |
| --- | --- |
| Always guess the most common token (".") | 7.6% |
| Count what followed the last token | 24.8% |
| Count what followed the last 4 tokens, falling back to fewer | 41.8% |
| Count what followed the last 8 tokens, falling back to fewer | 41.7% |
| The network, reading the last 8 tokens | 41.8% |

Split by whether the last 4 tokens appear together anywhere in the training stories:

| Test tokens | Share | Counting, last 4 | The network |
| --- | --- | --- | --- |
| The last 4 tokens were seen in training | 69.6% (15,131) | 45.2% | 43.9% |
| The last 4 tokens were never seen together | 30.4% (6,599) | 34.0% | 36.9% |

The same split by the last 8 tokens: on the 17.4% whose 8-token context was seen, counting gets 59.6% and the network 55.6%; on the other 82.6%, counting gets 37.9% and the network 38.9%.

What the table learned, in the finished network:

- **Neighbors** (cosine similarity of rows): "cat" is closest to "mouse" 0.87, "kitten" 0.84, "fluffy" 0.83 and "dog" 0.83. "big" sits next to "small" (0.80), and "up" next to "down" (0.80).
- **Word arithmetic.** Over 11 puzzles of the form b − a + c, the right answer comes first in 8 when the puzzle's three words are left out of the answers, and in 5 when they are not.
- **Groups.** Among the 42 map words, the average cosine within a group is 0.78 for colors, 0.69 for places, 0.66 for animals, 0.54 for feelings, 0.52 for people and 0.42 for actions. Between groups it is 0.14 to 0.20.

## From checkpoint to page

`tools/words/package.js` turns the final checkpoint into the two files the page loads.

- **`words/stories.js`** (5,643,995 bytes) holds the 1,922 merges, the training tokens and the 1,000 test stories' tokens. The tokens are stored as 16-bit little-endian numbers, gzipped, then base64. A header gives the source, the license and how the data was changed.
- **`words/weights.js`** (1,945,265 bytes) holds the network's 354,896 numbers rounded to 32 bits, which halves the download. Every number the page shows is measured with these rounded weights.
- **The saved moments** are the rows of the 42 map words at steps 0, 1,000, 3,000, 10,000, 40,000, 150,000 and 600,000 of the same run, for Secret 1 and the film. Step 0 is recomputed from seed 1.
- **The log** is the run's validation loss and accuracy at each checkpoint, for the film's training curve.

## Training in the reader's browser

Readers train with the same code in two places, in a background thread, without any server.

- **The thread.** The page builds a Web Worker from its own network script plus a small training loop, through a Blob URL. Where a page or browser blocks workers, the same code runs on the main thread instead.
- **Secret 1's replay.** Seed 1 and a learning rate of 0.001 give the same starting weights and the same batches as the finished network's run. It trains to step 1,000 (64,000 examples), saves the 42 map words' rows at steps 0, 250 and 1,000, and reports every 150 ms. That took about 30 to 36 seconds in Chrome on an Apple M5 Pro. Outside the browser, the page's one-thread trainer matched the finished run's table after 1,000 steps to 2 × 10⁻¹⁴. Browsers round a few math functions slightly differently, so their replays can differ in the last digits.
- **"Train your own" in the playground.** A fresh network starts from seed 2 (each Reset moves to the next seed) and trains at a constant rate of 0.001 until paused. The worker sends a copy of all the weights every 400 ms, which the guesses, the neighbors and the map then use. Every 500 steps, the page measures right first guesses on every 10th token of the 100 test stories (2,173 tokens), using a copy frozen at that step.
- **How far a reader gets.** Chrome on an M5 Pro trains about 2,000 examples a second. By the finished run's own curve, that is about 28% after 1 to 2 minutes, 33% after 5 and 36% after 10. The finished network's 38.4 million examples would take about 5 hours.

## Checks

| Check | How | Result |
| --- | --- | --- |
| Backward pass | the page compares 40 gradients of at least 0.001, rows of the table included, with central differences (h = 10⁻⁵) at every load | largest relative difference 10⁻⁷ |
| Python translation | `tools/words/pycheck.py` runs the page's Python on the same texts and the same three Adam steps | same tokens, merges and losses; weights within 1.1 × 10⁻¹² |
| Merges | re-learning them from the training words | all 1,922 reproduced |
| Multi-core trainer | 1 worker and 8 workers against the page's `trainStep`, 60 steps | bit-identical; 3.5 × 10⁻¹⁴ |
| Replay | the page's one-thread trainer against the finished run, 1,000 steps | 2 × 10⁻¹⁴ |
| Packaging | `stories.js` decoded against the token files; `weights.js` against the checkpoint | identical; equal up to the 32-bit rounding |
| Reviews | an independent accuracy review of every sentence on the page, and a code review | about 30 and 15 findings, all fixed |

**Retraining after a tokenizer fix.** The quote-direction step at first looked at the raw character before each quotation mark. That character could never be a curly quote, so a quote just inside another quote was misread. The fix (12 October 2026) looks at the last character written instead. It changed the tokens of one training story ("‘naughty’ means…", training story 1,424, which is story 1,425 of the file), left the token count at 4,300,373 and left the merges unchanged. The network was retrained with the same command, and every result matched the first run within 0.1 point.

## How to reproduce it

The commands run from `tools/words/` and need Node 24 or later; `pycheck.js` also needs Python 3. Downloads, token files, checkpoints and logs go to `tools/words/work/`, which git ignores.

```sh
cd tools/words
sh fetch.sh                  # the TinyStories files, about 170 MB, into work/dl/
node prep.js                 # merges and token files, a few seconds
mkdir -p work/runs
node ptrain.js k=8 h=128 lr=0.001 steps=600000 decay=0.3 workers=8 save=Q128 > work/runs/Q128.txt
                             # the finished network: about 54 minutes on 8 cores of an Apple M5 Pro
node package.js Q128 600000  # writes words/stories.js and words/weights.js
node pycheck.js              # the Python translation against the JavaScript
node baselines.js Q128_600000.bin   # counting and the network on the page's test tokens
node analyze.js Q128_600000.bin     # neighbors and word arithmetic
```

`node train.js` trains with the page's own `trainStep` on one thread, step for step like the page. `node phrases.js` shows what counting and the network expect after given phrases. `node look.js` opens the served page in headless Chrome and prints every live readout (serve the repo with `python3 -m http.server 8765`).

The same command on the same machine gives bit-identical weights. A different number of workers changes the order of additions, and other machines or JavaScript engines can round some math functions differently, so their weights differ in the last digits.
