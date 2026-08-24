# Roadmap

Where this is going: reading *Attention Is All You Need* and understanding
significant portions of it, from the bottom up, one broken module at a time.

The gap, stated plainly: **chapter 1 teaches neural networks in scalars, and
the paper is written entirely in matrices.** Every equation in it —
`Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` — is dense linear algebra. Chapter 2
exists to close that, and nothing after it makes sense until it is closed.

## Chapters

```
ch1_one_neuron    01-04   a neuron, a learning rule, two layers, gradient descent
ch2_arrays        05-07   matrices, the linear layer, and automatic differentiation
ch3_training      08-11   activations, softmax, optimizers, regularization
ch4_structure     12-13   convolution and recurrence
ch5_attention     14-19   embeddings, attention, and the transformer
```

`just chapters` prints what exists. `just chapter 2` runs one chapter,
`just run 7` runs one exercise wherever it lives.

## The modules

### ch1 — one neuron *(done)*

| | module | what it is for |
|---|---|---|
| 01 | `neuron` | weighted sum, threshold, and the decision boundary |
| 02 | `perceptron` | the first learning rule: nudge on every mistake |
| 03 | `network` | two layers, a sigmoid, backpropagation, and XOR |
| 04 | `gradient_descent` | what "nudge" actually means, with no network in the way |

### ch2 — arrays

| | module | functions to fix |
|---|---|---|
| 05 | `vectors` | `dot`, `matvec`, `matmul`, `transpose`, `add_bias`. The integration test re-expresses 03's forward pass as matrix operations and demands identical numbers |
| 06 | `linear_layer` | `linear_forward`, `grad_W`, `grad_x`, `grad_b`, checked against numerical differences |
| 07 | `autodiff` | local gradient rules and topological ordering. Rebuild 03's network on it and get the gradients you derived by hand in 03. After this you never derive one again, and you know what PyTorch is doing |

### ch3 — training

| | module | functions to fix |
|---|---|---|
| 08 | `activations` | `relu`, `relu_grad`, `tanh_grad`, `gelu`. The payoff is a plot of gradient magnitude by depth: why deep networks did not train until they did |
| 09 | `softmax` | `softmax`, `softmax_stable`, `cross_entropy`, and the gradient that collapses to `p - y` |
| 10 | `optimizers` | `xavier_scale`, `he_scale`, `sgd_step`, `momentum_step`, `adam_step`, `warmup_lr` — the last one is the paper's schedule formula, verbatim |
| 11 | `regularization` | `dropout_mask`, `l2_penalty`, `label_smooth`, and train-vs-validation curves |

### ch4 — structure

| | module | functions to fix |
|---|---|---|
| 12 | `convolution` | `conv1d`, `conv2d`, `pad`, `stride`, `max_pool`, `output_size`, `param_count`. Weight sharing, and why it costs so much less than dense |
| 13 | `recurrence` | `rnn_cell`, unrolling, backprop through time. The plot is gradient magnitude against distance back in time — the exact problem the paper exists to solve |

### ch5 — attention

| | module | functions to fix |
|---|---|---|
| 14 | `embeddings` | `one_hot`, `embed` (lookup as a matmul), weight tying, a minimal BPE merge |
| 15 | `attention` | `scores`, `scale_by_sqrt_dk`, softmax over scores, `weighted_sum(w, V)`, `causal_mask` — including *why* √d_k, which is the variance argument in the paper's footnote 4 |
| 16 | `multi_head` | `split_heads`, `merge_heads`, the Q/K/V projections, the output projection |
| 17 | `positional` | the sinusoid formula. First demonstrate that attention without it is permutation-invariant, so the need is felt rather than asserted |
| 18 | `block` | `layer_norm`, `residual`, `position_wise_ffn`, pre-LN vs post-LN. Stack N blocks and watch the gradients die without the residuals |
| 19 | `transformer` | assemble all of it, character-level, on a small corpus. Then read Figure 1 of the paper against your own file list |

## What unlocks what in the paper

| paper section | needs |
|---|---|
| §1 Introduction — why recurrence is a problem | 13 |
| §2 Background — comparison against convolution | 12, 13 |
| §3.1 Encoder and decoder stacks | 18 |
| §3.2.1 Scaled dot-product attention | 05, 09, 15 |
| §3.2.2 Multi-head attention | 16 |
| §3.3 Position-wise feed-forward | 06, 08 |
| §3.4 Embeddings and softmax | 09, 14 |
| §3.5 Positional encoding | 17 |
| §4 Why self-attention (the complexity table) | 12, 13 |
| §5.3 Optimizer and warmup schedule | 10 |
| §5.4 Regularization | 11 |

Safe to skip on a first read: §5.2 (hardware), the BLEU scores, beam search,
and §6.3 (constituency parsing). None of it is the architecture.

## Critical path

Not everything here is equally load-bearing.

- **The spine is 05, 06, 07, 09, 14, 15, 16, 17, 18, 19.** Skip any one of
  these and the paper has a hole in it.
- **08, 10 and 11** are not needed to parse the equations, but without them
  the paper's training section reads as incantation.
- **12 (convolution) is a genuine detour.** The paper's abstract defines
  itself as "dispensing with recurrence and convolutions entirely." Do it
  because convolution is worth knowing, not because transformers need it.
- **13 (recurrence) is not needed for the equations but is needed for the
  argument.** The whole motivation is that recurrence prevents parallelization
  within a training example. If you have never felt that, the paper's central
  claim lands as a fact rather than as a relief.

## A suggested order

```
05 → 06 → 07 → 09 → 08 → 15 → 16 → 17 → 18 → 14 → 19
```

with 10, 11, 12, 13 slotted in wherever a break is wanted. Attention arrives
early that way — right after softmax, which is its only real prerequisite —
rather than making you wait through all of the training machinery first.
