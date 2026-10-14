# Inside the Network

An interactive guide to neural networks that you can see and touch. Every figure is a live computation running in your browser. You don't need a GPU, a server or a machine-learning library.

**Live page:** https://taoyiran.github.io/inside-the-network/

## Chapter 1: A network you can touch

A two-input network with one hidden layer learns to tell blue dots from orange ones.

- **The film:** a 55-second animation that follows one dot through the forward pass, the loss, the backward pass and the update. It then fast-forwards 300 steps and explains the solution the network found, read from its trained weights.
- **The puzzle:** try to separate the dots with a single straight line, and see why one line is not enough.
- **The playground:** train the network yourself. Tap any dot, wire or neuron to see its numbers, step through training one stage at a time, and drag any weight by hand. Each gradient is shown as a chain-rule product and checked against autograd.

Everything runs on a ~40-line tape-based autograd written in plain JavaScript, in 64-bit floats with a fixed random seed, so every visitor starts from the same weights.

## Chapter 2: Loss and gradients, walking downhill

Why networks are trained on a loss instead of accuracy, what the loss measures, and what a gradient is: a neuron with two weights turns its whole loss into a landscape you can see, and gradient descent into a ball rolling downhill.

## Chapter 3: Backprop, one operation at a time

What happens inside the Backward click. A film follows one neuron's gradients backward operation by operation; three tools (local slopes, the chain rule, the tape) build up the algorithm; and a playground steps through computation graphs one operation per click, explaining each step in words, numbers and code, with a nudge test that checks every gradient the slow way.

## Chapter 4: Going deeper, why layers help

Why a network with two narrow hidden layers learns some shapes far faster than one wide layer with more parameters. A race on four rings of dots, a slider that moves each dot to the coordinates a layer gives it, every neuron of a trained network drawn as a square (lines, then curves, then shapes), the layer-by-layer backward pass in words, math, JavaScript and Python, and a measurement of vanishing gradients in a network with six hidden layers. The playground builds networks of up to three hidden layers.

## Chapter 5: Seeing, convolutional networks

A small convolutional network reads handwritten digits, and the chapter opens it up. A stencil you can edit slides over a digit, a layer-by-layer view shows every feature map, and three secrets follow. First, the stencils teach themselves: a fresh network trains in your browser and its first-layer stencils, started very small, turn from noise into edge detectors. Second, gradient ascent on the pixels shows what the network thinks each digit looks like. Third, a faint, aimed change to the pixels fools it. The playground lets you draw a digit, compare it with a plain network, move it around, see where the network looks, and train your own.

The digits are a 7,000-image subset of the [MNIST database](http://yann.lecun.com/exdb/mnist/) by Yann LeCun and Corinna Cortes, under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); `seeing/digits.js` is shared under that license, and so are `seeing/weights.js` (the two networks trained in advance on those digits with the chapter's own code) and images made from the digits.

## Chapter 5′: Listening, spectrograms and sound

A branch of chapter 5. Sound becomes a picture (a spectrogram), so chapter 5's convolutional network, with the same code, can recognize the spoken digits "zero" to "nine". The chapter opens with why recognizing sound matters and why it is hard, then takes a recording apart step by step (window, Fourier transform, mel bands, decibels). Four secrets follow: what the first-layer stencils learned (where sounds start and end, the lower edge of a band of sound); where the network listens; why moving a word in time barely matters while moving it in pitch does, until it is trained on pitch-shifted copies; and how noise hurts. A final section separates sounds by painting a mask on a spectrogram, the working space of many separation networks. The playground listens to your microphone; the sound never leaves the browser.

The recordings are 9,000 clips of spoken digits from the [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) by Pete Warden (Google), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); `listening/sounds.js` holds a subset, stored as spectrograms and 51 audio clips (16-bit and 8-bit μ-law). `listening/weights.js` holds the two networks trained in advance with the chapter's own code.

## Chapter 6: Words as vectors, tokens and embeddings

How a network reads text. The chapter opens with the problem, guessing the next word, and why it is hard: most stretches of text are new, and counting what followed a phrase only works for phrases already seen. Text is cut into tokens by byte-pair encoding (the 1,922 merges replay on a slider), each token gets a row of 32 numbers in an embedding table, and a small next-token network reads the last 8 tokens' rows. Four secrets follow: the rows of 42 words arrange themselves into groups over training, whose first steps replay in your browser; close means "used alike", which puts some opposites side by side; directions between rows can carry meaning, and analogy puzzles come out right more often when the puzzle's own words are left out of the answers; and every word has a single row, whatever it means in its sentence, which is what attention (chapter 7) fixes. The playground cuts your text into tokens, shows what comes next, lets the network write on, and trains your own network from scratch.

The stories are 21,000 from the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and Yuanzhi Li (Microsoft Research), under the [Community Data License Agreement – Sharing 1.0](https://cdla.dev/sharing-1-0/); `words/stories.js` holds them lowercased, tidied and cut into tokens, and is shared under that agreement. `words/weights.js` holds the network trained in advance with the chapter's own code. How it was trained, step by step, is in [`docs/chapter-6-training.md`](docs/chapter-6-training.md).

## Chapter 7: Attention, looking back over the whole story

How a network can use anything said earlier in a story, not just the last few tokens. The chapter opens with the problem: language keeps pointing back, and a window like chapter 6's cannot simply grow, while the right next word was often said long before (in the test stories, the hero's name comes back from further than 8 tokens on 1.6% of all tokens, and chapter 6's network almost never guesses it). A step-by-step animation then shows attention on a toy small enough to see every number: keys and a query drawn as arrows, their dot products, softmax, and the weighted average of the values, with a query you can drag. Attention is then built from three tools: a lookup by content (queries, keys and values, with each head's scores and weights shown for any token you click, and why a query–key score is a match the head has learned, not chapter 6's closeness of the words' rows), a learned score for each distance back so that order still counts, and chapter 6's hidden and output layers around it. A second animation places it in the transformer: from chapter 6's network to this one, to a transformer block (layer norm, attention, residual connection, MLP), a stack of blocks and a GPT, with real models' sizes and a glossary of the terms. With about as many weights as chapter 6's network, it guesses 46.6% of the test tokens right against 41.8%, and the story's name 45.3% of the time against 2.6%. Four secrets follow: the 8 heads found jobs during training (the token just before, words a few tokens back, names far back), shown at saved moments of training; it copies the names it knows, measured by swapping the story's name for others; switching a head off shows what each one is for, such as the one head that carries names; and one layer cannot finish an invented name made of several pieces, which takes two layers working together (an induction head, chapter 8). The playground shades your text by where any token looks, lets the network write on, and trains your own network from scratch.

The stories and the tokenizer are chapter 6's (`words/stories.js`). `attention/weights.js` holds the network trained in advance with the chapter's own code, shared under MIT like the code, with moments saved during its training. How it was trained, measured and checked is in [`docs/chapter-7-training.md`](docs/chapter-7-training.md).

## Run it locally

The chapters share files in `assets/`, and some pages read their own code from there to display it, which browsers block for pages opened straight from disk. Serve the folder instead:

```sh
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Project layout

- `index.html`, `loss-and-gradients/`, `backprop/`, `going-deeper/`, `seeing/`, `listening/`, `words/`, `attention/`: one folder per chapter, each a self-contained page.
- `assets/site.css`: the shared design tokens and components.
- `assets/autograd.js`: the shared autograd. Chapters 1 to 3 run on it, and chapter 4 checks its hand-written backward pass against it. Chapter 3 shows it in five parts, split at blank lines, next to a Python translation with the same five parts, so keep exactly five blank-line-separated sections in both.
- `assets/common.js`: shared helpers: number formatting, hover descriptions, the Words · Math · Code switch.
- `tools/`: the scripts that built each chapter's data and trained weights, and checked the numbers the pages state (see `tools/README.md`).
- `docs/`: detailed write-ups of how [chapter 6's](docs/chapter-6-training.md) and [chapter 7's](docs/chapter-7-training.md) networks are trained.

## Roadmap

1. [A network you can touch](https://taoyiran.github.io/inside-the-network/)
2. [Loss and gradients: walking downhill](https://taoyiran.github.io/inside-the-network/loss-and-gradients/)
3. [Backprop, one operation at a time](https://taoyiran.github.io/inside-the-network/backprop/)
4. [Going deeper: why layers help](https://taoyiran.github.io/inside-the-network/going-deeper/)
5. [Seeing: convolutional networks](https://taoyiran.github.io/inside-the-network/seeing/)
   - 5′. [Listening: spectrograms and sound](https://taoyiran.github.io/inside-the-network/listening/)
6. [Words as vectors: tokens and embeddings](https://taoyiran.github.io/inside-the-network/words/)
7. [Attention: looking back over the whole story](https://taoyiran.github.io/inside-the-network/attention/)
8. A tiny GPT, trained in your browser

## License

- Code: [MIT](LICENSE)
- Text and figures: [CC BY 4.0](LICENSE-CONTENT.md)
- Speech Commands recordings and spectrograms in `listening/sounds.js`, and the weights trained on them in `listening/weights.js`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), from the [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) by Pete Warden (Google)
- TinyStories in `words/stories.js` (lowercased, tidied and cut into tokens): [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/), from the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and Yuanzhi Li (Microsoft Research). The network trained on them, in `words/weights.js`, is a result of computation on the data, on which that agreement places no conditions (section 3.5); it is shared under MIT like the code.
- MNIST digits in `seeing/digits.js`, the weights in `seeing/weights.js` and images made from the digits: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), from the [MNIST database](http://yann.lecun.com/exdb/mnist/) by Yann LeCun and Corinna Cortes
