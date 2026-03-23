---
layout: distill
title: JAX-LM Assignment
description: The assignment companion for JAX-LM
tags: distill formatting
giscus_comments: true
date: 2026-03-20
featured: false
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Chuyi Shang
    url: "https://www.chuyishang.com"
    affiliations:
      name: UC Berkeley
  - name: Erin Tan
    url: "https://www.linkedin.com/in/erin-a-tan/"
    affiliations:
      name:
  - name: Rishi Athavale
    url: "https://www.linkedin.com/in/rishi-athavale-aba53a202/"
    affiliations:
      name:

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: "Getting Started"
  - name: "1. Basic Building Blocks: NNX Modules"
    subsections:
      - name: "1.1 Linear"
      - name: "1.2 Embedding"
      - name: "1.3 RMSNorm"
      - name: "1.4 Feed-Forward Network"
      - name: "1.5 Relative Positional Embeddings"
  - name: "2. Attention & Full Transformer Block"
    subsections:
      - name: "2.1 Scaled Dot-Product Attention"
      - name: "2.2 Causal Multi-Head Self-Attention"
      - name: "2.3 Transformer Block"
      - name: "2.4 Transformer LM"
  - name: "3. Training Loop"
    subsections:
      - name: "3.1 Cross Entropy Loss"
      - name: "3.2 Learning Rate Scheduler"
      - name: "3.3 Gradient Clipping"
      - name: "3.4 Training Loop"
  - name: "4. Sharding"

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

<!-- A complete implementation for building a basic Transformer-based language model in JAX is provided by us at: [https://github.com/chuyishang/cs336-jax](https://github.com/chuyishang/cs336-jax) -->
This assignment serves as a companion for the [JAX-LM](https://www.chuyishang.com/blog/2026/jax-lm/) blog. For the ambitious readers who want to take a stab at implementing our language model in JAX end-to-end, this assignment provides step-by-step instructions and a test suite to guide your implementation.

This assignment is built on top of [Assignment 1 (basics): Building a Transformer LM](https://github.com/stanford-cs336/assignment1-basics/tree/main) of [Stanford's CS336: Language Modeling From Scratch](https://cs336.stanford.edu/) course.

The main differences are:
1. We convert the test cases from PyTorch to JAX
2. We add tests and instructions for distributed training in JAX
3. Some sections are organized a little differently from the original assignment

However, the modules can mostly be completed in the same order.

{% details Device Requirements %}
It is strongly recommended to complete this assignment on a system with accelerators (GPU/TPU) to fully observe the speedups from JAX at scale.

That being said, this assignment is compatible with Linux, Windows, and MacOS systems, for sections up to and including **Section 3: Training Loop** can be completed and run locally on a standard laptop. This will still be helpful for just getting comfortable coding in JAX. However, the distributed  implementations assume access to GPUs/TPUs.

If you don't have acess to accelerators, you can simulate a device mesh using CPUs instead by setting `os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"`, where you can replace `8` with your number of desired devices. This must be done before JAX is imported. However, we have not fully tested this configuration. {% enddetails %}


## Getting Started
The blank starter code is provided on the `assignment-starter-code` branch. To clone only the starter code:
```bash
git clone --single-branch --branch assignment-starter-code https://github.com/chuyishang/cs336-jax.git
```

**Repository Structure**
The repository is a fork of `stanford-cs336/assignment1-basics`, and contains:
- `jax_tests/`: the core test suite for JAX-based implementations.
  - Suggested workflow: go through the sections below, passing all non-distributed tests (everything other than `test_sharding`). Then go back through to implement sharding.
  - `adapters.py`: This file starts as function stubs. You will edit it as you implement more components to get the relevant tests to run.
- `jax_impl/`: the directory where your code will live. Feel free to create files and organize your code as you see fit.
- `data/tokenizer.py` and `data/train_bpe.py`: BPE implementation and training script are provided by default because that code does not differ from the original PyTorch assignment.
- `uv.lock` and `pyproject.toml` which provide dependencies including JAX

By default, `uv run` and `uv sync` will assume a Linux + CUDA setup and be able to run out of the box. If you are using a CPU-only setup, or on a macOS device, you should first run:
```bash
uv sync --no-group cuda
``` 

**Running Tests**
Tests can be run using `uv run pytest jax_tests/test_file_name.py` or `uv run pytest -k test_file_name`. As a base, you should be able to run 
```bash
uv run pytest jax_tests/test_train_bpe.py jax_tests/test_tokenizer.py
```
and pass these tests right away since we've provided byte-pair tokenization implementations in `data/train_bpe.py` and `data/tokenizer.py`.

---
## 1. Basic Building Blocks: NNX Modules

> If you're following along on the CS336 spec, this section is the JAX-equivalent to **Section 3: Transformer Language Model Architecture**.
{: .block-tip }

In this section, we will build our first `nnx.Module` classes that will become the building blocks for our Transformer LM. Specifically, we will implement
- `Linear`
- `Embedding`
- `RMSNorm`
- `SwiGLU`
- `RoPE`: RoPE does not use random weight initialization

In your implementation, these modules can be defined anywhere under `jax_impl/` as long as `jax_tests/adapters.py` is updated to call them.

### Random Number Generation
NNX modules that require random weight initialization should be initialized with an additional `rngs` parameter. This is because in NNX, random number generation uses explicit states passed in as `nnx.Rngs` objects. 

Thus, your class signature should look something like:
```python
from flax import nnx
import jax.numpy as jnp

class Linear(nnx.Module):
	def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int, dtype: jnp.dtype=jnp.float32):
		pass
	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		pass
```

Another note is that the forward pass of an `nnx.Module` is implemented directly within the `__call__` method, instead of a separate `forward()` method like in PyTorch.

### Device Management
Recall that we also don't need to pass in devices explicitly. Later on, we will modify this section to support sharding annotations, but we don't need to worry about that now.

### 1.1 Linear
Let's get started with the first module: a linear layer. Again, the class signature for your module should look something like:
```python
class Linear(nnx.Module):
	def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int, dtype: jnp.dtype=jnp.float32):
		pass
	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		pass
```

The `Linear` module performs the following transformation:
$$y = xW$$
We first need need to initialize our weight matrix $W \in \mathbb R^{d_{\text{in}} \times d_{\text{out}}}$. Following the original assignment, we will initialize our weights to a truncated normal distribution $W \sim \mathcal N(\mu=0, \sigma^2 = \frac{2}{d_\text{in} + d_\text{out}})$ truncated at $[-3\sigma, 3\sigma]$. This is made easy thanks to [`nnx.initializers.truncated_normal`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.initializers.truncated_normal.html). You can also play around with more sophisticated [initializations](https://docs.jax.dev/en/latest/jax.nn.initializers.html) offered by the library, although it may not pass the autograder tests.

Once the weight data is initialized, it should be wrapped in an `nnx.Param` object and stored as a class attribute. 

When the function is called, we just apply the linear transformation.

> ##### TODOs
>
> 1. Implement the `Linear` module using only JAX arrays and methods.
>    - Note: don't forget to call the superclass constructor.
> 2. Update `run_linear` in `adapters.py` to call your method.
>    - Note: to pass the weights to your linear layer, call [`nnx.update`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.update) with the parameters formatted in a [`State`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/state.html#flax.nnx.State) dictionary. We do not include a bias term in the weights, and your implementation shouldn't either to work with the tests and adapter as is.
> 3. Run `uv run pytest -k test_linear` to check against the test case.
{: .block-tip }

> If you're stuck, we provide an in-depth walkthrough for implementing a Linear layer in [Implementing Our Model](/blog/2026/jax-lm/#implementing-our-model). We only do this for select sections. To just see a completed implementation of all sections, refer to the reference implementation!
{: .block-tip }

### 1.2 Embedding
Next up is the `Embedding` module, which takes as input a vector of integer token IDs and converts each of them to their vector representations. Its interface is very similar to `Linear`:
```python
class Embedding(nnx.Module):
	def __init__(self, rngs: nnx.Rngs, n_embeddings: int, embedding_dim: int, dtype: jnp.dtype=jnp.float32):
		pass
	def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
		pass
```

To make things a bit clearer:
- `n_embeddings` = vocabulary size
- We maintain a weight matrix $W \in \mathbb R^{\text{n\_embeddings} \times \text{embedding\_dim}}$ where the $i$-th row gives the embedding for token ID $i$. 
  - This can be initialized to $W \sim \mathcal N(0, 1)$ truncated at $[-3, 3]$ 
- The output will itself be a matrix of size $(\text{token\_ids.size}, \text{embedding\_dim})$ 

> ##### TODOs
>
> 1. Implement the `Embedding` module using only JAX arrays and methods.
> 2. Update `run_embedding` in `adapters.py` to call your method.
> 3. Run `uv run pytest -k test_embedding` to check against the test case.
{: .block-tip }

### 1.3 RMSNorm
`RMSNorm` is a basic layer normalization module. For convenience, the formula given a $d$-dimensional vector of activations as input:
```python
class RMSNorm(nnx.Module):
	def __init__(self, rngs: nnx.Rngs, d_model: int, eps: float = 1e-5, dtype: jnp.dtype=jnp.float32):
		pass
	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		pass
```

and the formula

$$
\text{RMSNorm}(x_{i}) = \frac{x_{i}}{\text{RMS}(x)} g_{i}
$$

where

$$
\text{RMS}(x) = \sqrt{\frac{1}{\text{d\_model}} \sum_{i=1}^{\text{d\_model}} a_{i}^2 + \epsilon}
$$

Here, our *gain* vector $g \in \mathbb R^{1 \times \text{d\_model}}$ consists of our learnable parameters and $\epsilon$ is a hyperparameter.

> ##### TODOs
>
> 1. Implement the `RMSNorm` module using only JAX arrays and methods.
> 2. Update `run_rmsnorm` in `adapters.py` to call your method.
> 3. Run `uv run pytest -k test_rmsnorm` to check against the test case.
{: .block-tip }

### 1.4 Feed-Forward Network
Now we get into implementing the actual feed-forward network that will be used as part our Transformer LM. The assignment calls for a [SwiGLU](https://arxiv.org/abs/2002.05202) (Swish-Gated Linear Unit) architecture, which consists of a Gated Linear Unit with a Swish (or SiLU) activation. Concretely:

$$
\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2(\text{SiLU}(W_1 x) \odot W_3 x)
$$

The dimensions here are:
- $x \in \mathbb R^{d_\text{model}}$
- $W_1, W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
$d_{\text{ff}}$ is the hidden size of the feed-forward block, while $d_\text{model}$ is the hidden size of the token representation. $d_{\text{ff}}$ should be set to approximately $\frac{8}{3} \cdot d_{\text{model}}$. 

The SiLU activation function is
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

> ##### TODOs
>
> 1. Implement the SiLU activation function.
>    - Update `adapters.run_silu` and call `uv run pytest -k test_silu` to test your implementation.
> 2. Implement the `SwiGLU` module using only JAX arrays and methods.
>    - Note: use your own `Linear` class!
> 3. Update `run_swiglu` in `adapters.py` to call your method.
> 4. Run `uv run pytest -k test_swiglu` to check against the test case.
{: .block-tip }

### 1.5 Relative Positional Embeddings
The last module for this section will implement [Rotary Positional Embeddings](https://arxiv.org/abs/2104.09864) (RoPE) to inject information about each token's relative position in the sequence. Here is the class interface:
```python
class RotaryPositionalEmbedding(nnx.Module):
	def __init__(self, d_k: int, theta: float, max_seq_len: int):
		pass
	def __call__(self, x: jnp.ndarray, token_positions: jnp.ndarray) -> jnp.ndarray:
		pass
```

Notably, this is the first module we've implemented that doesn't include `rngs` in the class signature. This is because this layer has no learnable parameters and thus does not need any randomized initialization.

Because of this, it's also possible to optimize RoPE by caching the sine and cosine values using [`nnx.Cache`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Cache) at initialization. 

**TODO: put the definition of RoPE**

> ##### TODOs
>
> 1. Implement the `RotaryPositionalEmbedding` module using only JAX arrays and methods.
> 2. Update `run_rope` in `adapters.py` to call your method.
> 3. Run `uv run pytest -k test_rope` to check against the test case.
{: .block-tip }

---
## 2. Attention & Full Transformer Block
The modules we have so far together form almost the entire Transformer, with the exception of the actual self-attention part. That's what we'll implement now.

In this section, we will implement 
- Scaled Dot-Product Attention
- Causal Multi-Head Self-Attention
- Transformer Block
- Transformer Language Model
### 2.1 Scaled Dot-Product Attention
The SDPA operation takes as input the query, key, and value matrices to perform the attention operation, defined as}""
$$\text{Attention}(Q, K, V) = \text{softmax}\bigg(\frac{Q K^\top}{\sqrt{ d_{k} }} \bigg) \cdot V$$
where $Q \in \mathbb R ^{n \times d_{k}}$, $K \in \mathbb R^{m \times d_{k}}$, and $V \in \mathbb R^{m \times d_{v}}$. Here, $m$ is the sequence length and $n$ is the batch size.

For convenience, the numerically-stable softmax activation function is:
$$\text{softmax}(x_{i}) = \frac{e^{x_{i} - \max(x)}}{\sum_{j} e^{x_{j} - \max(x)}}$$
> ##### TODOs
>
> 1. Implement the `softmax` activation function.
>    - Update `adapters.run_softmax` and call `uv run pytest -k test_softmax_matches_pytorch` to test your implementation.
> 2. Implement the `scaled_dot_product_attention` using only JAX arrays and methods.
>    - Note: include an optional `mask` parameter which accepts an array of shape `Float[Array, " ... queries keys"]`
> 3. Update `run_scaled_dot_product_attention` in `adapters.py` to call your method.
> 4. Run `uv run pytest -k test_4d_scaled_dot_product_attention` to check against the test case.
>
> *Hint: if you've done the original PyTorch assignment, these methods will look nearly identical.*
{: .block-tip }

### 2.2 Causal Multi-Head Self-Attention

We provide an in-depth walkthrough for implementing Causal Multi-Head Self-Attention in [Implementing Our Model](/blog/2026/jax-lm/#implementing-our-model). 

```python
class MultiHeadSelfAttention(nnx.Module):
	def __init__(
		self,
		rngs: nnx.Rngs,
		d_model: int,
		num_heads: int,
		rope_theta: float = 1e4,
		max_seq_len: int = 1024,
	):
		pass
	
	def __call__(self, x: Array, use_rope: bool = False):
		pass
```

> ##### TODOs
>
> 1. Implement the `MultiHeadSelfAttention` module using only JAX arrays and methods.
> 2. Update `run_multihead_self_attention` in `adapters.py` to call your method.
> 3. Run `uv run pytest -k test_multihead_self_attention` to check against the test cases.
{: .block-tip }

### 2.3 Transformer Block
The last piece of the puzzle is the Transformer block. Conveniently, we've already implemented all the submodules we'll need. Refer to Figure 2 of the below image for the architecture, taken from the original assignment:

{% include figure.liquid loading="eager" path="assets/img/jax-lm/transformer.jpg" class="img-fluid rounded z-depth-1" alt="transformer.png" zoomable=true %}

Here's the `TransformerBlock` interface:

```python
class TransformerBlock(nnx.Module):
	def __init__(
		self,
		rngs: nnx.Rngs,
		d_model: int,
		num_heads: int,
		d_ff: int,
		max_seq_len: int,
		theta: float,
	):
		pass
		
	def __call__(self, x: jnp.ndarray):
		pass
```

> ##### TODOs
>
> 1. Implement the `TransformerBlock` module using only JAX arrays and methods.
> 2. Update `run_transformer_block` in `adapters.py` to call your method.
> 3. Run `uv run pytest -k test_transformer_block` to check against the test case.
{: .block-tip }

### 2.4 Transformer LM
Now its finally time to put the pieces together to create the final Transformer language model. The structure is pasted in Figure 1 in the previous subsection. Similar to the Transformer block, we've already implemented all the layers we'll be needing.

```python
class TransformerLM(nnx.Module):
	def __init__(
		self,
		rngs: nnx.Rngs,
		d_model: int,
		num_heads: int,
		d_ff: int,
		theta: float,
		vocab_size: int,
		context_length: int,
		num_layers: int,
	):
		pass

	def __call__(self, x: jnp.ndarray):
		pass
```

> ##### TODOs
>
> 1. Implement the `TransformerLM` module using only JAX arrays and methods.
> 2. Update `run_transformer_lm` in `adapters.py` to call your method.
> 3. Run `uv run pytest -k test_transformer_lm` to check against the test cases.
{: .block-tip }

And there you have it: your own Transformer language model from scratch, completely in JAX. We'll upgrade this code in Section 4 to add sharding functionality, but this should've gotten you comfortable coding in JAX and NNX.

---
## 3. Training Loop
Now that we have all the pieces of the model, it's time to build out the training infrastructure. This will include:
- Cross entropy loss function
- Optax gradient transformations:
  - AdamW
  - Gradient clipping
  - LR schedule
- Train loop with checkpointing

### 3.1 Cross Entropy Loss
Cross entropy loss is defined as
$$\ell_{i}(o, x) = -\log \text{softmax}(o_{i})_{x_{i+1}}$$
where the Transformer outputs logits $o \in \mathbb R^{m \times \text{vocab size}}$ for each sequence $x$ of sequence length $m$. Then, the total cross entropy loss for a batch of size $B$ would be
$$\mathcal{L}(o, x)  
=  
-\frac{1}{B}\sum_{i=1}^{B}  
\log \big(\mathrm{softmax}(o_i)\big)_{x_i}$$
For numerical stability, you should implement the **log-sum-exp trick** (a helpful blog post about it [here](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)). 

*Note: the test cases can be used as a helpful reference for understanding the expected dimensions of the inputs.*

> ##### TODOs
>
> 1. Implement the cross entropy loss function.
> 2. Update `run_cross_entropy` in `adapters.py` to call your method.
> 3. Run `uv run pytest -k test_cross_entropy` to check against the test case.
>
> *Hint: if you've done the original PyTorch assignment, these methods will look nearly identical.*
{: .block-tip }

### 3.2 Optimizer
In JAX, the optimizer is treated as a series of Optax transformations. We can first construct our pipeline by adding gradient clipping followed by AdamW. Note that we pass in our learning rate schedule `lr_schedule` as an argument to AdamW during initialization. Then, we can create an `nnx.Optimizer` using our `model`, the pipeline we just created, and the `wrt` argument that specifies what to optimize.

This pipeline could look something like
```python
def build_optimizer_transform(
    optimizer_config: dict,
    gradient_clip: float,
    lr_schedule: optax.Schedule,
) -> optax.GradientTransformation:
    ...
```

> ##### TODOs
>
> 1. Implement the `get_lr_cosine_schedule` method for cosine annealing LR scheduling using only JAX arrays and methods.
> 2. Update `run_get_lr_cosine_schedule` in `adapters.py` to call your method.
> 3. Run `uv run pytest -k test_get_lr_cosine_schedule` to check against the test case.
{: .block-tip }


### 3.4 Training Loop

> An in-depth walkthrough for is provided in [Implementing the Training Loop](/blog/2026/jax-lm/#implementing-the-training-loop).
{: .block-tip }


---
## 4. Sharding
 
