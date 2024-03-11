---
    layout: ../../layouts/blogpost.astro
    title: Learning the Transformer
    pubDate: 2023-08-21
    description: "Learn how to implement the transformer from scratch using JAX"
    tags: ["transformer", "jax", "ai"]
---

# Learning the Transformer

2023-08-21

## Contents

## Introduction

The Transformer is one of the most successful neural network architectures in recent history that powers many of the new AI language models, such as ChatGPT, LLaMA, Claude and others. The transformer architecture was published in 2017 in the now famous Attention Is All You Need paper and in this blog post, we'll take a deep dive into the transformer architecture and try to implement it in Jax (although you can follow along with any other NN library). In this blog post, you will learn about the crux of the transformer: self-attention. So, grab a cup of coffee, pen and paper and let's get started!

You'll probably have to re-read this blog a couple of time to really understand the transformer. In a lot of places, you will see things like "this will be explained later", which can disrupt the flow of reading. Therefore, don't worry if you don't undestand it on the first try. Most people don't. Just keep re-reading until it clicks! 🙃

## Overview

Before we dive into the transformer, it's helpful to start with a high level overview first. In the following figure, you can see the transformer architecture.

<div class="figures">
    <img class="half" src="/posts/learning-the-transformer/Transformer.drawio.svg" />
    <img class="half" src="/posts/learning-the-transformer/Decoder.drawio.svg" />
</div>

The traditional transformer architecture consists of two parts: the encoder and the decoder each of which has a certain role. Let's start with the encoder's role, which you can interpret as the "listener". It takes as input the raw tokens and generates a fixed sized vector. The encoder is typically added in scenarios where the input data first needs to be encoded somehow, before the output tokens are generated. For instance, in translation tasks (such as the original transformer paper), you would get as input the English text "I am happy" and your goal is to translate this into German. Here, you would first pass the text through the encoder and get as output essentially a summary of the input tokens. The decoder would then take that summary as input during its computation. This is what I meant by the encoder being the "listener".

The decoder on the other hand takes the encoder's output (and we will see later exactly what this means) as well as its own output as input. This is what makes the transformer auto regressive: it takes its own output as input and generates new tokens continiously.

As you can see, there are many parts in the transformer, the most important being the Multi-Head Attention block. But we're getting ahead of ourselves. Let's start at the beginning.

## The Input

First, we have our input tokens. In machine learning, everything is a matrix, same with our input tokens. So what are the dimensions of this input matrix? In general, those are

$$
    X = [B \times T \times D]
$$

where $B$ is the batch size, $T$ is the number of tokens and $D$ is the vocabulary size. It can vary, depending on whatever tokeniser you used. If you follow along with Andrej Karparthy's transformer tutorial then $D$ is 65 (since he tokenises on the character level, while sentencepiece or tiktoken tokenise on the token-level, i.e. sub-word level). Let's have a look at some examples.

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
encoded = enc.encode("hello, world")

print(f"{encoded=}")
print(f"{enc.n_vocab=}")
encoded=[31373, 11, 995]
enc.n_vocab=50257
decoded = enc.decode(encoded)
print(f"{decoded=}")
decoded='hello, world'

```

in this example, $D$ is 50257. But it makes sense to extend this number to the nearest multiple of 64 for 2 reasons: first, it will make
$D$ divisible by 2, which will be important later and secondly, GPUs like multiples of 64 for efficiency reasons.

So, let's change the code a bit to extend $D$ to the nearest multiple of 64.

```python
def get_next_multiple_of_64(number: int) -> int:
    while number % 64 != 0:
        number += 1
    return number

n_vocab = get_next_multiple_of_64(enc.n_vocab)
```

Next up is the input embeddings, which is a simple embedding layer. You embed the input (which has dimensionality
$D$) into some - preferably smaller - dimension
$d$. Let's have a look at the code.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Float

class Transformer(eqx.Module):
    input_embedding: eqx.nn.Embedding

    def __init__(self, n_dims: int, n_embd: int, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 20) # let's just split 20 for now, we'll probably need them later
        self.input_embedding = eqx.nn.Embedding(n_dims, n_embd, key=subkeys[0])


key = jax.random.PRNGKey(42)
N_EMBD = 4096
transformer = Transformer(n_dims=n_vocab, n_embd=N_EMBD, key=key)
print(f"{transformer.input_embedding.weight.shape=}")
print(f"Number of bits = {transformer.input_embedding.weight.shape[0] * transformer.input_embedding.weight.shape[1] * 32}")
transformer.input_embedding.weight.shape=(50304, 4096)
Number of bits = 6593445888
```

As you can see, our input embedding is quite large. It holds 50304 _ 4096 _ 32 bits worth of information. That's around 824 megabytes, just for the input embedding.

Next up is the positional encoding. This one is a bit tricky since there are many different ways to compute the positional encoding (it can even be learned too). However, it's way more efficient to come up with some strategy which adds positional information to our inputs.

The authors used a sinusoidal positional encoding. But actually, anything goes, as long as you give each token a value that represents its position relative to all the other tokens. LLaMA, for example, uses a different kind of positional encoding, called "Rope Embeddings", but that's a story for another blog post. For now, let's implement the positional encoding:

```python
def get_positional_encoding(n_tokens: int, n_vocab: int) -> Float[Array, "n_tokens n_vocab"]:
    pos = jnp.arange(n_tokens)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, n_vocab, 2) * -(jnp.log(10000.0) / n_vocab))
    # alternatively: div_term = 1 / 10000 ** (jnp.arange(0, D, 2) / D)
    # that's closer to the actual notation they used.
    pos_enc = jnp.zeros((n_tokens, n_vocab))
    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(pos * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(pos * div_term))
    return pos_enc
```

If you had a lot of GPUs you could even learn the positional encodings directly by simply adding another embedding layer to our transformer. But given how much memory just the input embeddings took, it might not be the most efficient idea. In practice, both approaches yield similar results. And this is also precisely why we wanted to have even numbered dimensions earlier and why we extended
$D$ to the nearest multiple of 64 (making it an even number). We did that, because we want to apply
$\sin$ and
$\cos$ to every second dimension, meaning we needed an even number of dimensions to begin with.

Our input part is almost done. All that's left is to simply compute the positional encodings and add them to the input matrix, before feeding the input to the multi-head attention (MHA) block. We will come back to this part when we actually start training our transformer.

## Multi-Head Attention

TODO
