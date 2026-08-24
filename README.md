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
just run all       # every exercise, in order
just chapter 2     # every exercise in one chapter
just chapters      # what exists so far
just plot 1        # the figures for one exercise
```

The exercises live in chapter directories now but the numbering is global, so
`just run 7` finds exercise 7 without me having to remember which chapter it
ended up in.

## Who Wrote This Crap

I asked Claude to write these for me and I complained about them not being broken enough until the act of fixing them made the stuff start making more sense to me.

## Files

### `ch1_one_neuron/`

- `01_neuron.py` — one neuron, forward pass only. Weighted sum + threshold. No learning.
- `02_perceptron.py` — same neuron plus a learning rule, so it finds its own weights.
- `03_network.py` — two layers and a sigmoid. The one that can solve XOR.
- `04_gradient_descent.py` — what "nudge the weights" actually means, with no network in the way.

### `ch2_arrays/`

- `05_vectors.py` — dot, matmul, transpose. Chapter 1 in scalars; everything after this in matrices.

### and then

`ch3_training/`, `ch4_structure/`, `ch5_attention/` are empty so far.

## Where This Is Going

[ROADMAP.md](ROADMAP.md) — the plan for getting from here to reading
*Attention Is All You Need* and understanding a decent chunk of it. 19 modules,
5 chapters, with notes on which ones are actually load-bearing and which are
detours I want to take anyway.
