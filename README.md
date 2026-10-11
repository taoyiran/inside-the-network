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

## Run it locally

The chapters share files in `assets/`, and some pages read their own code from there to display it, which browsers block for pages opened straight from disk. Serve the folder instead:

```sh
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Project layout

- `index.html`, `loss-and-gradients/`, `backprop/`, `going-deeper/`: one folder per chapter, each a self-contained page.
- `assets/site.css`: the shared design tokens and components.
- `assets/autograd.js`: the one autograd every chapter runs. Chapter 3 shows it in five parts, split at blank lines, next to a Python translation with the same five parts, so keep exactly five blank-line-separated sections in both.
- `assets/common.js`: shared helpers: number formatting, hover descriptions, the Words · Math · Code switch.

## Roadmap

1. [A network you can touch](https://taoyiran.github.io/inside-the-network/)
2. [Loss and gradients: walking downhill](https://taoyiran.github.io/inside-the-network/loss-and-gradients/)
3. [Backprop, one operation at a time](https://taoyiran.github.io/inside-the-network/backprop/)
4. [Going deeper: why layers help](https://taoyiran.github.io/inside-the-network/going-deeper/)
5. Seeing: convolutional networks
   - 5′. Listening: spectrograms and sound
6. Words as vectors: tokens and embeddings
7. Attention
8. A tiny GPT, trained in your browser

## License

- Code: [MIT](LICENSE)
- Text and figures: [CC BY 4.0](LICENSE-CONTENT.md)
