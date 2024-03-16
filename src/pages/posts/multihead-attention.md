---
    layout: ../../layouts/blogpost.astro
    title: Multihead-Attention Deepdive
    pubDate: 2023-09-21
    description: "Take a deep dive into the Multihead-Attention block of the transformer"
    tags: ["transformer", "attention"]
---

# Multihead-Attention Deepdive

2023-09-21

## Contents

## Introduction

Multihead-Attention (MHA) is the key part of the transformer architecture. It uses a mechanism called self attention, which has been very successful in NLP tasks. I already introduced the transformer in my other blog post. In this blog post, we will take a deepdive into the MHA and try to make it as general as we can. Let's start with an overview of the MHA block. In the following figure, you can see the structure of the MHA block.

![MHA Block](/posts/multihead-attention/MHA.drawio.svg)

The input to the MHA block is duplicated into 3 vectors: the query, key and value vectors. Each of those is first passed through their respective linear layer. Each of those layers has a specific task:

- Query Layer: transforms the input to query vectors, i.e. what you're interested in
- Key Layer: transforms the input to a set of keys to match the query vectors against
- Value Layer: take the scaled combination of the query and key projects and compute the output of the MHA block

This is a good starting point, so let's write this down in code.

One thing to note is the dimensionalities of the vectors, so let's start by defining the dimensions first. Here's the notation for this blog post:

- $L$: maximum sequence length
- $h$: number of heads
- $\{q,k,v\}_{\text{dim}}$: query, key or value embedding dimension

Furthermore, let's define the input to the MHA block:

- Query: $[L \times q_{\text{in}}]$, where $q_{\text{in}}$ is the query input dimension
- Key: $[L \times k_{\text{in}}]$, where $k_{\text{in}}$ is the key input dimension
- Value $[L \times v_{\text{in}}]$, where $v_{\text{in}}$ is the value input dimension

Usually, the query, key and value input dimensions are the same, but we want our implementation to be very general and make as few assumptions as possible about the current use case. Therefore we will be more specific. The reason that normally they are the same is that, typically, they come out of the input embeddings (with the positional embeddings added on top) and the same embedding is used for all vectors, giving them all the same input dimension. Let's start with some imports and define the dimensions.

```python

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


query_input_dim = 16
query_embedding_dim = 32
key_input_dim = 16
key_embedding_dim = 32
value_input_dim = 16
value_embedding_dim = 32
num_heads = 4
max_seq_len = 10
batch_size = 2
key = jax.random.PRNGKey(42)
```

## Linear Layers

As shown in the MHA figure, we will first implement the linear layer parts.

```python

# Version 1
class MultiheadAttention(eqx.Module):
    query_projection: eqx.nn.Linear
    key_projection: eqx.nn.Linear
    value_projection: eqx.nn.Linear

    query_input_dim: int = eqx.field(static=True)
    query_embedding_dim: int = eqx.field(static=True)

    key_input_dim: int = eqx.field(static=True)
    key_embedding_dim: int = eqx.field(static=True)

    value_input_dim: int = eqx.field(static=True)
    value_embedding_dim: int = eqx.field(static=True)

    num_heads: int = eqx.field(static=True)

    def __init__(self, query_embedding_dim, key_embedding_dim, value_embedding_dim, query_input_dim, key_input_dim, value_input_dim, num_heads, key):
        qkey, kkey, vkey = jax.random.split(key, 3)
        self.query_projection = eqx.nn.Linear(query_input_dim, num_heads * query_embedding_dim, key=qkey, use_bias=False)
        self.key_projection = eqx.nn.Linear(key_input_dim, num_heads * key_embedding_dim, key=kkey, use_bias=False)
        self.value_projection = eqx.nn.Linear(value_input_dim, num_heads * value_embedding_dim, key=vkey, use_bias=False)

        # parameters
        self.query_input_dim = query_input_dim
        self.query_embedding_dim = query_embedding_dim
        self.key_input_dim = key_input_dim
        self.key_embedding_dim = key_embedding_dim
        self.value_input_dim = value_input_dim
        self.value_embedding_dim = value_embedding_dim
        self.num_heads = num_heads

    def __call__(self, x: Float[Array, "max_seq_len input_dim"]):
        seq_len, _ = x.shape
        query = jax.vmap(self.query_projection)(x).reshape(seq_len, self.num_heads, self.query_embedding_dim)
        key = jax.vmap(self.key_projection)(x).reshape(seq_len, self.num_heads, self.key_embedding_dim)
        value = jax.vmap(self.value_projection)(x).reshape(seq_len, self.num_heads, self.value_embedding_dim)
        print(f"{query.shape=}")
        print(f"{key.shape=}")
        print(f"{value.shape=}")

key, subkey = jax.random.split(key)
mha = MultiheadAttention(query_embedding_dim, key_embedding_dim, value_embedding_dim, query_input_dim, key_input_dim, value_input_dim, num_heads, key)
x = jax.random.normal(subkey, (max_seq_len, query_input_dim))
mha(x)
```

```
query.shape=(10, 4, 32) # L x h x q_embd
key.shape=(10, 4, 32) # L x h x k_emdb
value.shape=(10, 4, 32) # L x h x v_emdb
```

As mentioned in my previous blog post, a MHA block consists of multiple heads. But, instead of looping over each head, one at a time, we can instead simply enlarge the query, key and value layers to include all of the heads. Look at it this way: taking all of the heads into consideration, the output shape of, say, the query projection should be:

$$

[L \times h \times q_{\text{embd}}]


$$

By making the query projection layer project from $q_{n}$ to $h * q_{emb}$, we get initially a matrix with the same $[L \times h * q_{emb}]$. From there, we can simply reshape that matrix into our desired shape: $[L \times h \times q_{emb}]$. This is just the first steps for the query and key projections; they have still quite the journey ahead. We still need to matrix multiply, scale (sometimes mask) and softmax them. Let's write a function that can do all of that in one go.

```python

def dot_product_attention(query_projection: Array, key_projection: Array, value_projection: Array, mask: Optional[Array | None]) -> Array:
    attention_weights = jax.vmap(
        lambda q, k: q @ k.T,
        in_axes=(1, 1),
        out_axes=1
    )(query_projection, key_projection)
    attention_weights = attention_weights / jnp.sqrt(key_projection.shape[-1])
    attention_weights = jax.nn.softmax(attention_weights, axis=-1)

    return attention_weights
```

Masking is missing from the above implementation for now, but that's ok. We will add that in later.

Let's think for a moment what we even did in the function above and what our goal even was. First, we took the query and the key projections, i.e. the matrices that came after plugging our initial input through the linear layers. The task of the linear layers is to learn how to transform the input into usable queries and keys, such that they can be matched against each other.

Let's just imagine for a moment that our linear layers do a perfect job of transforming the input into the perfect queries and keys. What do we even do with those? This is where the self-attention mechanism comes into play.

## Self-Attention

To understand self-attention, we first need to understand "regular" attention. Here's a useful example. Think of YouTube and how you might want to find a particular video. You enter a query into the search bar. Now, how does YouTube know, which video to show you? Each video has a video title, a description and some other metadata. Those are the keys. Your query is matched against the keys and whichever has the highest value (according to whatever metric was using in the matching-process), is shown to you first.

![Attention](/posts/multihead-attention/Attention1.drawio.svg)

In that example, the query and keys are different. The query was what you searched for and the keys whatever YouTube has in its database. In self-attention on the other hand, the query and key are the same - at least initially.

Take the input sentence: "Hi, how are". That sentence is both the query and the key, albeit in its raw form. Matching those against each other directly isn't really useful - not to mention that it's always the same. We need to turn these raw query and keys into usable queries and keys, such that you can actually match one against the other. How we can do that specifically is unknown. We can, however, learn it. And that's precisely the goal of the linear layers.

![Self-Attention](/posts/multihead-attention/Attention2.drawio.svg)

Alright, let's continue with the MHA block, while keeping in mind, that we will add masking last. In our image we see that the next step is the matrix multiplication of the scaled attention weights with the value matrix.

```python

class MultiheadAttention(eqx.Module):
    ...
    def __call__(self, x: Float[Array, "max_seq_len input_dim"]):
        seq_len, _ = x.shape
        query = jax.vmap(self.query_projection)(x).reshape(seq_len, self.num_heads, self.query_embedding_dim)
        key = jax.vmap(self.key_projection)(x).reshape(seq_len, self.num_heads, self.key_embedding_dim)
        value = jax.vmap(self.value_projection)(x).reshape(seq_len, self.num_heads, self.value_embedding_dim)

        scaled_attention_weights = dot_product_attention(query, key, value, None)

        qkv_matmul = jax.vmap(
            lambda attention_weights, value: attention_weights @ value,
            in_axes=(1, 1),
            out_axes=1
        )(scaled_attention_weights, value)
```

As you can see, we compute the matrix multiplication between the attention weights and the value. We use jax.vmap to map over axis 1 of the inputs. Just as a reminder, the shape of the attention weights and the values are:

$$

[L \times h \times h] \text{ and } [L \times h \times v_{\text{embd}}]
$$

where the first is the shape of the attention weight matrix and the latter is the shape of the value projection and $h$ is the number of heads. Why is the shape of the attention weights $L\times h \times L$? We mentioned earlier how in self-attention the query and the key are both the input (that went through the linear layers, but from now on, whenever I reference the query and keys you can assume that I'm referring to the ones after having their linear layers applied to them). When we compute the how much attention each word gives to another word in the sequence, then it only makes sense that we have a square matrix! If your query has 3 words, then you have 3 keys; matching each query with each key gives you a 3x3 matrix. From there, we have to remind ourselves, that we have a multihead attention block, which means that each head - individually and independently - computes the attention weights. Since we want to use matrices to leverage efficient GPU usage, we keep everything in a tensor and simply add the head dimension in the middle. It's in the middle simply for consistency, as we chose the middle dimension (i.e. axis 1) to be the number of heads.

By defining `jax.vmap(..., in_axes=(1, 1), out_axes=1)` we tell Jax to vectorise across axes `(1, 1)` for the input (which is $h$) and then we define `out_axes=1`, which puts that axes back into the output in axis position 1. The following figure illustrates this.

![Vmap](/posts/multihead-attention/VMAP-MHA.drawio.svg)

Now, we need to concatenate the heads and then apply the final linear output layer. We're still missing the output layer in our MHA implementation, so first let's define the output's dimension:

```python

output_dim = 32
```

And let's add the output to our current implementation, both in the `__init__` and the `__call__` functions.

```python

class MultiheadAttention(eqx.Module):
    ...
    output: eqx.nn.Linear
    output_dim: int = eqx.field(static=True)

    def __init__(self, query_embedding_dim, key_embedding_dim, value_embedding_dim, query_input_dim, key_input_dim, value_input_dim, num_heads, output_dim, key):
        qkey, kkey, vkey, okey = jax.random.split(key, 4)
        ...
        self.output = eqx.nn.Linear(num_heads * value_embedding_dim, output_dim, key=okey, use_bias=False)

        # parameters
        ...
        self.output_dim = output_dim

    def __call__(self, x: Float[Array, "max_seq_len input_dim"]):
        ...
        concatenation = qkv_matmul.reshape(seq_len, -1)
        ...
        output = jax.vmap(self.output)(concatenation)
        return output

key, subkey = jax.random.split(key)
mha = MultiheadAttention(query_embedding_dim, key_embedding_dim, value_embedding_dim, query_input_dim, key_input_dim, value_input_dim, num_heads, output_dim, key)
x = jax.random.normal(subkey, (max_seq_len, query_input_dim))
output = mha(x)
```

```
ic| query.shape: (10, 4, 32)

ic| key.shape: (10, 4, 32)
ic| value.shape: (10, 4, 32)
ic| attention_weights.shape: (10, 4, 10)
ic| qkv_matmul.shape: (10, 4, 32)
ic| concatenation.shape: (10, 128)
ic| output.shape: (10, 32)
```

We can see that the output shape is $L \times q_{embd}$, but it needn't necessarily be; it's just usually the case.

Our implementation is already ok-ish flexible, but we can go one step further, namely by looking at the number of heads! So far, we just assumed that each head should have 1 query, key and value, but this need not be the case! In fact, there is something called Multi-Query Attention (MQA), where one key and value matrix is shared across all heads. Personally, I would have gone with the name Single-Key-Value Attention (SKVA) just to highlight that it's the keys and the values that change.

Let's review. Our current implementation (the standard MHA block) has hh heads and each head has a query, key and value. This means in MHA, we have $h$ queries, keys and values. In SKVA (i.e. MQA) we have $h$ heads and $h$ queries but only 1 key and value. So, let's go one step further and define Multi-Key-Value Attention (MKVA), where we will have $h$ heads, $h$ queries and $d$ keys and values.

By the way, in this case, MHA would have d = h and in SKVA (MQA) we would have d = 1.

Here's the difference between SKVA (MQA) and MHA:

<div class="figures">
    <img class="half" src="/posts/multihead-attention/StandardMHA.drawio.svg" />
    <img class="half" src="/posts/multihead-attention/SKVA.drawio.svg" />
</div>

As you can see, key 1 and value 1 are shared across all heads. Now, let's think for a moment what MKVA would look like. Here's a first draft:

![MKVA](/posts/multihead-attention/MKVA.drawio.svg)

We're treading into untapped waters here. Implementations from the big players such as PyTorch or TensorFlow aren't as customisable, which means I couldn't find a reference implementation.

Let's start simple and give these dimensions a name: `key_multihead_dim`, `value_multihead_dim` and `query_multihead_dim`, although I'm not sure if we will even need last one at all. Also, it might very well be that what I'm trying to do here is not even possible, but let's try it anyway. First, let's add these dimensions to the `__init__` method.

```python

...
key_multihead_dim = 4
value_multihead_dim = 4
query_multihead_dim = 4
...

class MultiheadAttention(eqx.Module):
    query_multihead_dim: int = eqx.field(static=True)
    key_multihead_dim: int = eqx.field(static=True)
    value_multihead_dim: int = eqx.field(static=True)

    def __init__(self, ..., query_multihead_dim, key_multihead_dim, value_multihead_dim, key):
        ...
        # parameters
        ...
        self.query_multihead_dim = query_multihead_dim
        self.key_multihead_dim = key_multihead_dim
        self.value_multihead_dim = value_multihead_dim
```

We need to come up with a solid plan of attack for our generic implementation; most importantly, we need to figure out what we actually want to achieve, which is harder to articulate than you might think. Basically, if you set e.g. key_multihead_dim to any number lower than the number of heads, it effectively means that not all heads will have a key. For example, if your MHA block has 8 heads and only a single shared key, i.e. `key_multihead_dim = 1`, then all heads "share" that one key (SKVA/MQA). But if you set it to, say, `key_multihead_dim=7`, then 7 heads have a key and one does not. But in matrix multiplication you can't simply "skip" a dimension, which is going to be our main challenge.

The current plan of attack is to use a double vmap strategy, which I call the outer and inner vmap. The following figure shows a rough outline of my strategy.

![MKVA](/posts/multihead-attention/MKVANew.drawio.svg)

The rough idea is to have kind-of a nested for-loop. The outer `vmap` maps over axis 1 of the keys and values while the inner `vmap` maps over axis 1 of the queries (which is usually the number of heads). Upon closer inspection it would probably make more sense to have a single value for the `multihead_dim`, since - at the very least - the key and value dimensions must be the same (otherwise, we'd need a third vmap for `value_multihead_dim`). Actually, let's do that right now.

```python

...
kv_multihead_dim = 4
...

class MultiheadAttention(eqx.Module):
    ...
    kv_multihead_dim: int = eqx.field(static=True)
    ...
    def __init__(self, ..., query_multihead_dim, kv_multihead_dim, key):
        ...
        # parameters
        ...
        self.query_multihead_dim = query_multihead_dim
        self.kv_multihead_dim = kv_multihead_dim

    def __call__(self, x: Float[Array, "max_seq_len input_dim"]):
        ...

...
mha = MultiheadAttention(..., query_embedding_dim, kv_multihead_dim, key)
...

```

After performing the MKVA, the `out_axis` is placed (per default) on `out_axes=0` and we'd perform a summation over that out_axes and then compute the mean over the out_axes.

Doing it this way, is actually not 100% mathematically equivalent to what we had before! As it turns out, these vmaps are not exactly the same:

```python
vmap(fn, in_axes=(1, 1, 1))
vmap(vmap(fn, in_axes=(None, 1, 1)), in_axes=(1, None, None))
```

For now, we will keep it and later train a transformer with our MHA implementation to experimentally check if our new approach even works at all.

Let's implement our new strategy. First, we will update the `dot_product_attention` function to compute the `qkv_matmul` matrix directly using the value projections, while also keeping in mind our new matrix dimensions, which I have added to the function signature:

```python
def dot_product_attention(
    query_projection: Float[Array, "max_seq_len query_embedding_dim"],
    key_projection: Float[Array, "max_seq_len key_embedding_dim"],
    value_projection: Float[Array, "max_seq_len value_embedding_dim"],
    mask: Optional[Array | None],
) -> Array:
    ic(query_projection.shape, key_projection.shape, value_projection.shape)

    attention_weights = query_projection @ key_projection.T
    attention_weights = attention_weights / jnp.sqrt(key_projection.shape[-1])
    attention_weights = jax.nn.softmax(attention_weights, axis=-1)
    ic(attention_weights.shape)

    qkv_matmul = attention_weights @ value_projection

    return qkv_matmul
```

Here is the part that will perform the inner `vmap`:

```python

def vmapped_attention(query_heads, key_heads, value_heads, mask):
    attn_fn = ft.partial(dot_product_attention, mask=mask)
    ic(query_heads.shape, key_heads.shape, value_heads.shape)
    # Inner VMAP
    dpa = jax.vmap(
        lambda q, k, v: attn_fn(q, k, v),
        in_axes=(1, None, None),
        out_axes=1,
    )(query_heads, key_heads, value_heads)
    return dpa
```

Let's also adjust our MHA implementation to use the double `vmap` strategy:

```python
class MultiheadAttention(eqx.Module):
    ...

    def __call__(self, x: Float[Array, "max_seq_len input_dim"]):
        ...
        pt_vmapped_fn = ft.partial(
            vmapped_attention,
            mask=None,
        )


        # Outer VMAP
        qkv_matmul = jax.vmap(
            pt_vmapped_fn,
            in_axes=(None, 1, 1),
        )(query, key, value)

        qkv_matmul = jnp.sum(qkv_matmul, axis=0)

        # Taking the mean over the d dimension
        qkv_matmul = qkv_matmul / self.kv_multihead_dim


        concatenation = qkv_matmul.reshape(seq_len, -1)
        output = jax.vmap(self.output)(concatenation)

        return output

key, subkey = jax.random.split(key)
mha = MultiheadAttention(query_embedding_dim, key_embedding_dim, value_embedding_dim, query_input_dim, key_input_dim, value_input_dim, num_heads, output_dim, query_embedding_dim, kv_multihead_dim, key)
x = jax.random.normal(subkey, (max_seq_len, query_input_dim))
output = mha(x)
```

```

ic| query.shape: (10, 4, 32)
    key.shape: (10, 4, 32)
    value.shape: (10, 4, 32)
ic| query_heads.shape: (10, 4, 32)
    key_heads.shape: (10, 32)
    value_heads.shape: (10, 32)
ic| query_projection.shape: (10, 32)
    key_projection.shape: (10, 32)
    value_projection.shape: (10, 32)
ic| attention_weights.shape: (10, 10)
ic| qkv_matmul.shape: (10, 4, 32)
ic| concatenation.shape: (10, 128)
ic| output.shape: (10, 32)
```

As you can see, the output shape matches exactly what we had before.

The next step is to add masking, which we can simply copy from the other blog post. If we build a transformer around this MHA implementation and test our implementation on the TinyShakespeare dataset, then (at least in my preliminary tests) it doesn't seem to work! I'm not exactly sure why, but I suspect that it has something to do with the vmap strategy. I will investigate this further in the future and publish my findings in another post.
