# neural-networks-study

Here I am learning how neural networks work, by fixing broken ones that Claude made for me.

My goals here are **extremely modest**: I just want to grasp basic concepts I never understood before.

This is not intended to be anything useful to anyone else, but maybe someone will find it useful...¯\_(ツ)_/¯

## What Is This Crap

Each file is deliberately broken or half-built. I am going for "productive feailure" here.

Also, I am putting my solutions into a branch where I can go back to them.

## How To Run

I run them like this:

```sh
just run 1

```

Can also do:

```sh
just run all

```

## Who Wrote This Crap

I asked Claude to write these for me and I complained about them not being broken enough until the act of fixing them made the stuff start making more sense to me.

## Files

- `01_neuron.py` — one neuron, forward pass only. Weighted sum + threshold. No learning.
- `02_perceptron.py` — same neuron plus a learning rule, so it finds its own weights.
- `03_network.py` — two layers and a sigmoid. The one that can solve XOR.
