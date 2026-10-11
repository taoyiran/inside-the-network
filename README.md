# Inside the Network

An interactive guide to neural networks that you can see and touch. Every figure is a live computation running in your browser. You don't need a GPU, a server or a machine-learning library.

**Live page:** https://taoyiran.github.io/inside-the-network/

## Chapter 1: A network you can touch

A two-input network with one hidden layer learns to tell blue dots from orange ones.

- **The film:** a 55-second animation that follows one dot through the forward pass, the loss, the backward pass and the update. It then fast-forwards 300 steps and explains the solution the network found, read from its trained weights.
- **The puzzle:** try to separate the dots with a single straight line, and see why one line is not enough.
- **The playground:** train the network yourself. Tap any dot, wire or neuron to see its numbers, step through training one stage at a time, and drag any weight by hand. Each gradient is shown as a chain-rule product and checked against autograd.

Everything runs on a ~40-line tape-based autograd written in plain JavaScript, in 64-bit floats with a fixed random seed, so every visitor sees the same numbers.

## Run it locally

Open `index.html` in a browser. To serve it locally instead:

```sh
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Roadmap

1. A network you can touch *(this chapter)*
2. Backprop, one operation at a time
3. Going deeper: why layers help
4. Seeing: convolutional networks
   - 4′. Listening: spectrograms and sound
5. Words as vectors: tokens and embeddings
6. Attention
7. A tiny GPT, trained in your browser

## License

- Code: [MIT](LICENSE)
- Text and figures: [CC BY 4.0](LICENSE-CONTENT.md)
