# Build tools

The scripts that made each chapter's data and trained weights, and that checked the numbers the pages state. Pages that only train live in the browser (chapters 1 to 4) need none.

Everything runs with Node (24 or later) and, for a few scripts, Python 3 with no extra packages. Downloads, intermediate files, checkpoints and logs stay out of git: chapters 6 and 7 keep them in `words/work/` and `attention/work/`, and chapters 5 and 5′ next to their scripts (`data/`, `dl/`, `*.json`), all listed in the repo's `.gitignore`.

`python3 review_copy.py <out dir> [words/index.html …]` makes single-file copies of chapter pages, with the shared files and the chapter's data inlined, for a private review page.

## `words/`: chapter 6, Words as vectors

These scripts read the tokenizer and the network straight from `words/index.html` (`<script id="tok">` and `<script id="nwm">`), so they always run the page's own code. The whole training process, with every formula, setting and result, is written up in [`docs/chapter-6-training.md`](../docs/chapter-6-training.md).

1. `sh fetch.sh`: downloads the TinyStories V2 (GPT-4) validation file and the first 150 MB of its training file into `work/dl/`.
2. `node prep.js`: learns the 1,922 merges from training stories 2 to 20,001 and cuts the stories into tokens: `work/merges.json`, `work/train.bin`, `work/test.bin` (the first 2,000 complete validation stories; the page uses the first 1,000, and stories 1,001 to 2,000 are only for choosing among trained networks).
3. `node ptrain.js k=8 h=128 lr=0.001 steps=600000 decay=0.3 workers=8 save=Q128 > work/runs/Q128.txt`: trains the finished network, saving checkpoints to `work/models/`. Each step's 64 examples are shared out among worker threads; with `workers=1` the result is bit-identical to the page's own `trainStep`, and with 8 workers only the order of the additions differs. About an hour on 8 cores. `node train.js` is the same training on one thread, with the page's `trainStep` itself.
4. `node package.js Q128 600000`: writes `words/stories.js` and `words/weights.js`.

Checks:

- `node pycheck.js`: the page's Python translation against its JavaScript (tokens, merges, losses, weights after three Adam steps).
- `node baselines.js Q128_600000.bin`: counting what followed the last 1, 4 or 8 tokens, and the network, on every token of the first 100 test stories, split by whether the last 4 tokens appear in the training stories. The page measures the same live.
- `node analyze.js Q128_600000.bin`: nearest words and analogy puzzles. `phrases.js`: what counting and the network expect after given phrases.
- `node look.js [dark] [w=390]`: opens the served page in headless Chrome (`python3 -m http.server 8765` from the repo), prints every live readout and console message, and saves screenshots to `work/shots/`.

How the settings were chosen (October 2026, on validation stories): Adam learned far faster than plain gradient descent (35% against 32% right first guesses after 1.3 million examples); a table shared with the output layer, as in GPT-2, learned worse; 256 or 512 hidden neurons learned no faster than 128; dropout of 0.2 slowed learning without helping by the end; reading 16 tokens instead of 8 helped by under half a point. A plain 8-token network with 128 neurons was kept.

## `attention/`: chapter 7, Attention

These scripts read the attention network straight from `attention/index.html` (`<script id="attn">` and its Python translation `<script id="attn-py">`), and use chapter 6's tokenizer, network and token files through `../words/lib.js`, so chapter 6's `fetch.sh` and `prep.js` must have run first. The whole process, with every formula, setting and result, is written up in [`docs/chapter-7-training.md`](../docs/chapter-7-training.md).

1. `node ptrain.js lr=0.008 steps=19000 save=F8 > work/runs/F8.txt`: trains the finished network, 8 whole stories per step, saving checkpoints to `work/models/`. All the positions of a step's stories are shared out evenly among 16 worker threads (a share can start mid-story); with `workers=1` the result is bit-identical to the page's own `trainStep` (`node train.js`). About 33 minutes on an Apple M5 Pro.
2. `node package.js F8 19000`: writes `attention/weights.js`, with the run's log and the moments of training for Secret 1.

Checks: `verify.js` (the shipped weights against the checkpoint, the saved moments against fresh measurements, and the page's copy of the tokenizer against chapter 6's), `gradcheck.js` (the backward pass against central differences), `pycheck.js` (the Python translation against the JavaScript). The numbers the page states live: `compare.js` (counting, chapter 6 and attention, by where the answer last appeared), `names.js` (names far back, swapped and invented names, over the 1,000 test stories), `ablate.js` (heads switched off; `heads=R`: distance scores set to 0), `heads.js` (where each head looks), `swapnames.js` (Secret 2, per new name), `pieces.js` (Secret 4, piece by piece). `sweep.sh` reruns the settings sweep. `templates.js` scored candidate stories for the page's question; `pronouns.js` is a measurement that did not make it onto the page. `look.js` and `interact.js` open the served page in headless Chrome and print its live readouts.

## `seeing/`: chapter 5, Seeing

The scripts as they were used when the chapter was built, with paths changed to this folder. `cnn.js` and `model.py` are copies of the page's network code at the time; `pagecode5.js` loads the code from the live page instead.

1. Download the four MNIST files (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`; the chapter used the copy at `https://storage.googleapis.com/cvdf-datasets/mnist/`) into a folder, say `dl/`.
2. `mkdir -p data && python3 subset.py dl data && gzip -k data/train.bin data/test.bin`: the first 600 of each digit from the training set and the first 100 of each from the test set.
3. `node train_ref.js` (the convolutional network: `ref_weights.json`) and `node mlp_ref.js` (the plain networks: `mlp8_weights.json`, `mlp32_weights.json`).
4. `python3 package.py . ../..` (this folder, holding `data/` and the weights, then the repo): writes `seeing/digits.js` and `seeing/weights.js`.

Checks: `verify5.js` (the shipped weights match the trained ones, the page's starting weights and trainer reproduce the reference network, and the accuracies and the shift, fooling and dream numbers the page states), `gc_page.js` (the backward pass against the slow way), `xcheck_js.js` then `xcheck_py.py` (the Python translation), `secrets.js` and `secrets2.js` (the numbers behind the chapter's secrets).

## `listening/`: chapter 5′, Listening

Also as used, with paths changed. `audio.js`, `audio.py` and `init.js` are copies of the page's code at the time; `pageinit_node.js` runs them with chapter 5's network code from `../seeing/cnn.js`, and `pagecode5p.js` loads the code from the live page instead.

1. `python3 fetch.py dl`: streams the Speech Commands v0.02 archive and keeps the ten digit folders and the official split lists.
2. `mkdir -p data && node prep.js`: picks 800 training clips per digit (from neither official list) and 100 test clips per digit (from the official testing list), one clip per speaker: `data/pick.json`. `node loudness.js` reports the loudness used to choose the page's QUIET floor.
3. `node feats.js`: the 32 × 32 spectrograms, with the page's own function: `data/train.bin`, `data/test.bin`.
4. `node train_page.js cnn`, `node train_page.js plain`, `node train_page.js shifted`: the three networks, with the page's own trainer.
5. `node package.js ../..`: writes `listening/sounds.js` and `listening/weights.js`.

Checks: `verify5p.js` (the shipped files match, and the page's trainer reproduces all three networks bit for bit), `numbers.js` (accuracies, including the noise figures the page says were measured in advance), `secrets.js` and `sep.js` (the secrets and the separation figure), `train_ref.js` (an earlier version of the training, through chapter 5's harness), and in `checks/`: `snr.js` (how the hiss level compares with the word), `noisetrain.js` (training with hiss makes the network more robust) and `causal.js` (training on pitch-shifted copies makes the network shrug off pitch, and training on time-aligned pictures takes away its ease with time: the effect comes from the training data).
