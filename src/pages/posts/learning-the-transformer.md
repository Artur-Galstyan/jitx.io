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

Okay, now we get to the most interesting part and to the heart of the transformer. In the earlier figures I had just shown an overview where MHA was just a single box, but there's a lot going on in that box. So let's zoom in a bit.

![Multi-Head Attention](/posts/learning-the-transformer/MHA.drawio.svg)

Before we dive deep into the how and why, let's first identify our trainable parameters. As you can see, there are 4 linear layers: 3 for the query, key and value (whatever those are, we don't know just yet) and another one at the end. Let's implement a first draft and figure out the rest later.

```python
class MultiHeadAttention(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    output: eqx.nn.Linear
    def __init__(self, *, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 4)
        self.query = eqx.nn.Linear(key=subkeys[0], use_bias=False)
        self.key = eqx.nn.Linear(key=subkeys[1], use_bias=False)
        self.value = eqx.nn.Linear(key=subkeys[2], use_bias=False)
        self.output = eqx.nn.Linear(key=subkeys[3], use_bias=False)
```

We will come back and fix the above code in just a minute, but at this point it's time for an insertion!

## Insertion: What's the difference between "normal" attention and self attention?

A good question! Attention in machine learning is nothing new and has been used plenty before. However, attention was often calculated in relation to another query, like a question for example. Take a look at the following example:

Query: Doughnuts are tasty! What's the capital of Germany?

A: The capital of Germany is Berlin.

In this example, in the answer the attention of the words capital and Germany are high, as they coincide with the query a lot! In fact, if you think about it, there is little to no need to pay attention to the doughnut part. As you can see, in this example, attention is computed relative to some other query.

In self attention, on the other hand, the attention scores are calculated within the sentence itself!

![Self Attention](/posts/learning-the-transformer/SelfAttention.drawio.svg)

## Multi-Head Attention (again)

Okay, so what's up with these query, key and value vectors? The idea for those comes from recommender systems. Think about YouTube. What you search for is the query, for example: funny cats. YouTube has a huge database of multiple keys such as video title, description etc. Let's say there are only 3 videos in total on YouTube with the following titles:

- cats walking
- horses running
- dogs jumping

Your query is matched against all of these keys to see which one fits best. Let's say, their recommender systems computes these values:

- cats walking - 0.7
- horses running - 0.1
- dogs jumping - 0.2

As you can see, the first video has the highest value. The one with the highest value is the one you care most about, i.e. pay most attention to.

In self attention, we don't have any external keys (otherwise we would have "normal" attention). In our case, the input itself is both query and key! This is what people mean by self attention. The query, key and value vector you saw in the figure above: they're all the same (check circle with the number 1)! The input (after it had the positional encodings added to it) is duplicated 3 times and each copy is named the query, key and value. Let's say the input is the text "How are you". Both query and key contain the phrase "How are you" (in the form of numbers though, but we will get to that). We pass those vectors through the linear layers and then matrix multiply those results. The goal of the query linear layer is to learn what parts of the query tokens each token should pay most attention to (2). The key linear layer learns what tokens are most suitable to answer to the query tokens (2). The matrix multiplication of these matrices yields a score matrix, which holds the self attention information. In our simplified example:

![Self Attention Example](/posts/learning-the-transformer/SASimple.drawio.svg)

The result of the matrix multiplication of query and key can lead to very large numbers, which is why after the multiplication the result is scaled down, which also allows for more stable gradients later. Typically, the results are scaled down relative to the chosen dimensionality. More specifically, the scaling is:

$$
    \frac{\text{scores}}{\sqrt{d_k}},
$$

where $d_k$ is the dimension of the keys.

Note, how there are 3 sizes: D, which is the original input dimension, i.e. the vocabulary size. Then there's d, which is the dimension we chose for the input embedding, i.e. we embed D->d. Then there's d_k, which is d // n_heads, i.e. the embedding dimension divided n_heads times.

When we apply a softmax afterwords, we turn the attention weights into probabilities. At the end, we multiply the attention weights (i.e. query @ key) with the value vector. The idea is that the attention weights highlight the important parts in the value vector and diminish that which we should not pay any attention to.

So far, we have only taked about a single attention head. In a transformer, MHAs have multiple heads (hence the name). This means the input is not split once into query, key and value, but rather
$n_{\text{heads}}$ times (3). Oh no! Another parameter! Quick, let's add that to our implementation!

```python
class MultiHeadAttention(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(self, input_dim: int, n_heads: int, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 4)
        qkv_size = input_dim // n_heads
        self.query = eqx.nn.Linear(
            in_features=input_dim,
            out_features=qkv_size * n_heads,
            key=subkeys[0],
            use_bias=False
        )
        self.key = eqx.nn.Linear(
            in_features=input_dim,
            out_features=qkv_size * n_heads,
            key=subkeys[1],
            use_bias=False
        )
        self.value = eqx.nn.Linear(
            in_features=input_dim,
            out_features=qkv_size * n_heads,
            key=subkeys[2],
            use_bias=False
        )
        self.output = eqx.nn.Linear(
            in_features=input_dim,
            out_features=input_dim,
            key=subkeys[3],
            use_bias=False
        )
```

Uff, quite a lot changed, so let's have a look. We first added the number of heads as a parameter to the init function. But what's up with qkv_size? Well, we need to split the input dimension into the number of heads. (Remember, that the input dimension of the MHA block is not the same as the original dimension, but rather whatever we chose as the embedding dimension for the input embeddings.) That way, each head gets the chance to learn something new for its dimension, independently of the other heads, which are all busy with their respective slice of the input dimension. As you can see, the output layer has the same in and out feature dimension (4). The reason for that is the "concatenate" block you saw earlier, which concatenates all the outputs of the single attention heads into a single matrix, which has the same shape as the original input (which was split n_heads times). So why is there a qkv_size \* n_heads for each of the first 3 linear layers? Think of it this way: alternatively, you could have created a list of those linear layers and iterated each head one at a time. But that's a huge waste of computation! Instead, it's much better to just simply enlarge the linear layers and then - at the end - reshape the matrices to our desired size. We will see that in just a bit.

Alright, let's implement the MHA block (without masking just yet) and walk through all the parts.

```python
class MultiHeadAttention(eqx.Module):
    n_heads: int = eqx.field(static=True)
    qkv_size: int = eqx.field(static=True)

    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear

    output: eqx.nn.Linear
    def __init__(self, input_dim: int, n_heads: int, key: PRNGKeyArray) -> None:
        # input_dim will be what we chose for n_embd in the input embedding
        key, *subkeys = jax.random.split(key, 5)
        self.qkv_size = input_dim // n_heads
        self.query = eqx.nn.Linear(
            in_features=input_dim,
            out_features=qkv_size * n_heads,
            key=subkeys[0],
            use_bias=False
        )
        self.key = eqx.nn.Linear(
            in_features=input_dim,
            out_features=qkv_size * n_heads,
            key=subkeys[1],
            use_bias=False
        )
        self.value = eqx.nn.Linear(
            in_features=input_dim,
            out_features=qkv_size * n_heads,
            key=subkeys[2],
            use_bias=False
        )
        self.output = eqx.nn.Linear(
            in_features=input_dim,
            out_features=input_dim,
            key=subkeys[3],
            use_bias=False
        )

        self.n_heads = n_heads

    def _project(self, proj, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.n_heads, -1)

    def __call__(self, x: Array):
        T, _ = x.shape

        q = self._project(self.query, x)
        k = self._project(self.key, x)
        v = self._project(self.value, x)


        dot_product_vmap = jax.vmap(
            lambda q, k: jnp.dot(q, k.T),
            in_axes=(1, 1),
            out_axes=1
        )
        attention_scores = dot_product_vmap(q, k)
        # TODO: apply masking if needed - we'll get to this later
        attention_scores = attention_scores / jnp.sqrt(self.qkv_size)
        attention_scores = jax.nn.softmax(attention_scores)
        matmul_vmap = jax.vmap(
            lambda s, v: jnp.dot(s, v),
            in_axes=(1, 1),
            out_axes=1
        )

        output = matmul_vmap(attention_scores, v)
        output = output.reshape(T, -1)
        output = jax.vmap(self.output)(output)

        return output
```

Oh, that's quite a lot. Let's break all that down, step-by-step. The first couple of lines are Equinox specific ways to indicate static fields of the module, which are not meant to be trained during backpropagation later. In other libraries, you'll have to check how to specify static fields (in PyTorch, you can just assign those to `self`).

```python
    # this is Equinox specific
    n_heads: int = eqx.field(static=True)
    qkv_size: int = eqx.field(static=True)

```

Then, we have the `__init__` method:

```python
    def __init__(self, input_dim: int, n_heads: int, key: PRNGKeyArray) -> None:
        # input_dim will be what we chose for n_embd in the input embedding
        key, *subkeys = jax.random.split(key, 5)
        self.qkv_size = input_dim // n_heads
        self.query = eqx.nn.Linear(
            in_features=input_dim,
            out_features=qkv_size * n_heads,
            key=subkeys[0],
            use_bias=False
        )
        self.key = eqx.nn.Linear(
            in_features=input_dim,
            out_features=qkv_size * n_heads,
            key=subkeys[1],
            use_bias=False
        )
        self.value = eqx.nn.Linear(
            in_features=input_dim,
            out_features=qkv_size * n_heads,
            key=subkeys[2],
            use_bias=False
        )
        self.output = eqx.nn.Linear(
            in_features=input_dim,
            out_features=input_dim,
            key=subkeys[3],
            use_bias=False
        )

        self.n_heads = n_heads
```

I mentioned earlier that - since we have
$n$ heads - we need to split the input. Currently the input has the shape of
$[T\times d]$, where
$T$ is the number of tokens and
$d$ is the number of dimensions of the input (which - since the input will come after the embedding dimension, will be n_embd). However, after applying the linear layers, we want the query, key and value matrices to have the shape
$[T \times h\times d_k]$, where $h$ is the number of heads and $d_k$
is equal to $D/h$, i.e. the number of dimensions of the input divided by the number of heads. This is another reason why we wanted the dimension of the input embeddings earlier to be the nearest multiple of 64: if we make n_heads equal to a multiple of 32 for example, we are guaranteed an even divisio$n$ for $d_k$.
