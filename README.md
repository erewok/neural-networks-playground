# neural-networks-study

Here I am learning how neural networks actually work, by fixing broken ones that Claude made for me.

This is meant to be anything useful to anyone else, but if it's helpful...¯\_(ツ)_/¯

## The idea

Each file is deliberately broken or half-built. I am going for "productive feailure" here.

Also, I am putting my solutions into a branch where I can go back to them.

## Running

    just run 1
    just run 2
    just run 3
    just run 3 handwired    # extra args pass through
    just all

## Files

- `01_neuron.py` — one neuron, forward pass only. Weighted sum + threshold. No learning.
- `02_perceptron.py` — same neuron plus a learning rule, so it finds its own weights.
- `03_network.py` — two layers and a sigmoid. The one that can solve XOR.
