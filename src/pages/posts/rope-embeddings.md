---
    layout: ../../layouts/blogpost.astro
    title: RoPE Embeddings
    pubDate: 2023-10-19
    description: "RoPE embeddings is a new technique to represent positional embeddings in a transformer. In this post, you will learn the math and how to implement them."
    tags: ["transformer", "positional-embeddings", "rope"]
---

# RoPE Embeddings

2023-10-19

## Contents

## Introduction

Self-attention in a transformer is position-invariant, meaning that it the attention matrix has the same values regardless of the order of the words. Here's an example and notice how the values (which are made up by the way) are the same just at different positions.

![Position Invariance](/posts/rope-embeddings/PosInvar.drawio.svg)

This is a problem because the order of the words in a sentence is important. For example, "I like cats" and "cats like I" have the same words but different meanings and we wish to capture that. This is why, in the original Transformer paper, the authors added positional embeddings to the input, namely the sinusoidal embeddings. Here's a code snippet, which calculates those positional embeddings:

```python
# standard positional encoding
def get_positional_encoding(
    n_tokens: int, n_vocab: int
) -> Float[Array, "n_tokens n_vocab"]:
    pos = jnp.arange(n_tokens)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, n_vocab, 2) * -(jnp.log(10000.0) / n_vocab))
    # the following expression is closer to the actual notation they used.
    # div_term = 1 / 10000 ** (jnp.arange(0, n_vocab, 2) / n_vocab)
    pos_enc = jnp.zeros((n_tokens, n_vocab))
    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(pos * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(pos * div_term))
    return pos_enc
```

Once you have those positional encodings, you add them to the input. The core idea is to generate some unique position vector for each word in the sentence and add it to the word embedding. This way, even if the words are the same, the position vectors will be different and thus attention matrix will be different. In the case of sinusoidal embeddings, you can visualize it as generating lots of sine & cosine waves with different frequencies and stack those up. Then, for each position, you draw a that passes through all sine waves and measure the value for each wave. That process would look like this:

![Sinusoidal Embeddings](/posts/rope-embeddings/Sinus1.drawio.svg)

While it does give each word a unique position, it doesn't do it in a predictable way nor does it keep the relative distances between the words constant. We can easily visualise that by looking at the word vectors as we apply the sinusoidal embeddings:

![Sinusoidal Embeddings](/posts/rope-embeddings/sinusanimation.avifs)

As you can see, the arrow jumps around and the relative distances change. It's hard to predict where the arrow will jump to next.

## RoPE Embeddings

This is where we learn about RoPE embeddings. RoPE stands for Rotary Positional Embeddings and was published in the [RoFormer paper](https://arxiv.org/pdf/2104.09864.pdf). The core idea of RoPE embeddings is to use a rotation matrix to rotate the input vector. This way, the relative distances between the words are preserved and the position vectors are predictable. Take a look at the vector animation again, but this time with RoPE embeddings. Notice, how the arrow moves in a predictable way.

![RoPE Embeddings](/posts/rope-embeddings/ropeanimation.avifs)

Much better, right? In this blog post, we will implement a `RoPEEmbeddingsLayer` in Equinox and hopefully get that merged into the main repository.

## The Math

The core idea is very simple. Once you have your query and key vectors (the ones you
get from passing your input through the linear layers), you rotate them some angle
$m\theta$, where $m$ is the token position and $\theta$ is the rotation angle. For example, if you
have 3 tokens, then $m$ would be 0, 1, and 2. To rotate a vector, you multiply it by the
rotation matrix:

$$
\begin{bmatrix}\cos\theta & -\sin\theta\\\sin\theta & \cos\theta\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}x\cos\theta-y\sin\theta\\x\sin\theta+y\cos\theta\end{bmatrix}
$$

I'm not 100% sure why we apply the rotations after applying the linear weights. My guess is that if we applied them before the linear layers, then we would just distort the input vectors and the rotation effect would be lost. I'm not sure though.

To calculate the rotation angle, we use the following formula:

$$
\theta_i = 10000^{-\frac{2(i-1)}{d}},
$$

where $d$ is the dimensionality of the tokens (in the example above it was 2 but in practice it's a much bigger number). To extend this to the multidimensional case where we have more than just 2 dimensions, we would simply construct a sparse rotation matrix for each dimension and split the input vector into chunks of 2 and rotate each chunk separately. But that's a super slow operation and in practice we use this formula:

$$
R_{\theta,m}^d\mathbf{x} = \begin{pmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4 \\
\vdots \\
x_{d-1} \\
x_d
\end{pmatrix} \otimes \begin{pmatrix}
\cos(m\theta_1) \\
\cos(m\theta_1) \\
\cos(m\theta_2) \\
\cos(m\theta_2) \\
\vdots \\
\cos(m\theta_{d/2}) \\
\cos(m\theta_{d/2})
\end{pmatrix} + \begin{pmatrix}
-x_2 \\
x_1 \\
-x_4 \\
x_3 \\
\vdots \\
-x_{d-1} \\
x_d
\end{pmatrix} \otimes \begin{pmatrix}
\sin(m\theta_1) \\
\sin(m\theta_1) \\
\sin(m\theta_2) \\
\sin(m\theta_2) \\
\vdots \\
\sin(m\theta_{d/2}) \\
\sin(m\theta_{d/2})
\end{pmatrix}
$$

So, let's implement this! There are 2 ways to implement this: the real-numbers way and the complex-numbers way. Let's start with the former.

## The Real-Numbers Way

The $\cos$ and $\sin$ parts can be precomputed and cached since they don't change at all. As mentioned before, the formula for the rotation angle is:

$$
\theta_i = 10000^{-\frac{2(i-1)}{d}}
$$

```python
def get_freqs(seq_len: int, dim: int, base: int = 10000):
    theta = 1 / (base ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(seq_len)

    idx_theta = jnp.einsum("i,j->ij", t, theta)
    idx_theta = jnp.concatenate([idx_theta, idx_theta], axis=1)

    freqs_cos = jnp.cos(idx_theta)
    freqs_sin = jnp.sin(idx_theta)

    return freqs_cos, freqs_sin
```

The code above simply calculates all the indices first (i.e.
$m\theta$) and then simply applies
$\cos$ and
$\sin$ to each of those. The next step is to negate every other element in the input vector
$x$ before applying the frequencies.

```python
def get_rope_embedding(x: Float[Array, "seq_len dim"]):
    def negate_half(x: Float[Array, "seq_len dim"]):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([x[..., :d_2], -x[..., d_2:]], axis=-1)

    seq_len, dim = x.shape
    cos, sin = get_freqs(seq_len, dim)
    neg_half_x = negate_half(x)
    x_rope = (x * cos) + (neg_half_x * sin)
    return x_rope
```

That's really all there is to it.

## The Complex-Numbers Way

This is a bit more complicated but it's also faster if you JIT the code. Notice, in the function above, we return `freqs_cos`, `freqs_sin` as 2 separate arrays. As it turns out, we can express both of them as complex numbers:

```python
def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> Complex[Array, "end dim/2"]:
    def polar(abs, angle):
        return jnp.array(
            abs * jnp.cos(angle) + abs * jnp.sin(angle) * 1j, dtype=jnp.complex64
        )
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[jnp.newaxis, :] / dim))
    t = jnp.arange(end)
    idx_theta = jnp.outer(t, freqs)
    # this is the same as the following (if you're more of the einsum person)
    # idx_theta = jnp.einsum("i,j->ij", t.ravel(), freqs.ravel())
    freqs_cis = polar(jnp.ones_like(idx_theta), idx_theta)
    return freqs_cis
```

In the code above, we store the
$\cos$ values in the real part and the
$\sin$ values in the imaginary part. With the precomputed frequencies, we can now implement the RoPE embeddings:

```python
def get_rope_embeddings_complex(x: Float[Array, "seq_len dim"]):
    def negate_half(x: Float[Array, "seq_len dim"]):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([x[..., :d_2], -x[..., d_2:]], axis=-1)

    seq_len, dim = x.shape
    freqs = precompute_freqs_cis(dim, seq_len)
    neg_half_x = negate_half(x)
    freqs_real = jnp.tile(freqs.real, (1, 2))
    freqs_imag = jnp.tile(freqs.imag, (1, 2))

    x_rope = (x * freqs_real) + (neg_half_x * freqs_imag)
    return x_rope
```

We can verify that both implementations yield the same result with a simple test:

```python
import jax.numpy as jnp
from icecream import ic


dim = 1024
max_seq_len = 4096

x_rope_real = get_rope_embedding(jnp.ones((max_seq_len, dim)))
x_rope_complex = get_rope_embeddings_complex(jnp.ones((max_seq_len, dim)))

ic(x_rope_real)
ic(x_rope_complex)

ic(jnp.allclose(x_rope_real, x_rope_complex))

ic| x_rope_real: Array([[ 1.        ,  1.        ,  1.        , ...,  1.        ,
                          1.        ,  1.        ],
                        [ 1.3817734 ,  1.3869226 ,  1.3915513 , ...,  0.99989444,
                          0.99989635,  0.9998982 ],
                        [ 0.49315056,  0.5400876 ,  0.58551943, ...,  0.9997889 ,
                          0.9997927 ,  0.9997964 ],
                        ...,
                        [-0.40462857, -0.581694  , -0.2302509 , ...,  0.48944828,
                          0.49965236,  0.5096456 ],
                        [-1.3588928 ,  0.74914414, -1.2779073 , ...,  0.48930827,
                          0.4995152 ,  0.50951135],
                        [-1.0637972 ,  1.4135625 , -1.2257419 , ...,  0.4891682 ,
                          0.499378  ,  0.509377  ]], dtype=float32)
ic| x_rope_complex: Array([[ 1.        ,  1.        ,  1.        , ...,  1.        ,
                             1.        ,  1.        ],
                           [ 1.3817734 ,  1.3869226 ,  1.3915513 , ...,  0.99989444,
                             0.99989635,  0.9998982 ],
                           [ 0.49315056,  0.5400876 ,  0.58551943, ...,  0.9997889 ,
                             0.9997927 ,  0.9997964 ],
                           ...,
                           [-0.40462857, -0.581694  , -0.2302509 , ...,  0.48944828,
                             0.49965236,  0.5096456 ],
                           [-1.3588928 ,  0.74914414, -1.2779073 , ...,  0.48930827,
                             0.4995152 ,  0.50951135],
                           [-1.0637972 ,  1.4135625 , -1.2257419 , ...,  0.4891682 ,
                             0.499378  ,  0.509377  ]], dtype=float32)

ic| jnp.allclose(x_rope_real, x_rope_complex): Array(True, dtype=bool)
```

## Performance Comparison

Now, you might be wondering which implementation is faster. Let's find out by benchmarking them:

```python
dim = 1024
max_seq_len = 4096

test_cases = 1_000_0

x = jax.random.uniform(
    jax.random.PRNGKey(42), (max_seq_len, dim), dtype=jnp.float32
)
start_time = time.time()
for _ in range(test_cases):
    x_rope_real = get_rope_embedding(x)
end_time = time.time()
ic("Real:", end_time - start_time)
ic("Avg:", (end_time - start_time) / test_cases)
start_time = time.time()
for _ in range(test_cases):
    x_rope_complex = get_rope_embeddings_complex(x)
end_time = time.time()
ic("Complex: ", end_time - start_time)
ic("Avg:", (end_time - start_time) / test_cases)
```

When we run this, we get the following results:

|               | Complex  | Real     |
| ------------- | -------- | -------- |
| No JIT        | 299.1253 | 193.1669 |
| Avg. No JIT   | 0.029912 | 0.01931  |
| With JIT      | 71.4320  | 83.8433  |
| Avg. with JIT | 0.0071   | 0.0083   |

Interestingly, the real-numbers way is faster than the complex-numbers way when we don't JIT the code. However, once we JIT the code, the complex-numbers way is faster. We will go with the second implementation given the pretense that - when using JAX - you should always strive to JIT your code.

## Equinox Implementation

Let's write the RoPE embeddings for the Equinox library. Our implementation will have the constraint that the input's sequence length must remain the same, which makes our code less dynamic. The reason for this is that JAX - in general - does not like dynamic shapes at all, since when the shapes change, functions need to be recompiled. Here's the implementation:

```python
class RotaryPositionalEmbedding(eqx.nn.Module):
    dim: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)
    freqs_cis: Complex[Array, "dim/2"] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.freqs_cis = self.precompute_freqs_cis(dim, max_seq_len)

    @staticmethod
    def negate_half(x: Float[Array, "max_seq_len dim"]):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([x[..., :d_2], -x[..., d_2:]], axis=-1)

    @staticmethod
    def precompute_freqs_cis(
        dim: int, end: int, theta: float = 10000.0
    ) -> Complex[Array, "end dim/2"]:
        def polar(abs, angle):
            return jnp.array(
                abs * jnp.cos(angle) + abs * jnp.sin(angle) * 1j, dtype=jnp.complex64
            )

        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[jnp.newaxis, :] / dim))
        t = jnp.arange(end)
        freqs_outer = jnp.outer(t, freqs)
        freqs_cis = polar(jnp.ones_like(freqs_outer), freqs_outer)
        return freqs_cis

    @jax.named_scope("eqx.nn.RotaryPositionalEmbedding")
    def __call__(
        self, x: Float[Array, "max_seq_len dim"], *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "max_seq_len dim"]:
        max_seq_len, dim = x.shape
        assert (
            dim == self.dim
        ), f"x.shape[-1] must match self.dim, but {x.shape[-1]} != {self.dim}"
        assert dim % 2 == 0, f"x.shape[-1] must be even, but {x.shape[-1]} is not even."
        assert (
            max_seq_len == self.max_seq_len
        ), f"x.shape[0] must be == self.max_seq_len, but {x.shape[0]} != {self.max_seq_len}"
        neg_half_x = self.negate_half(x)
        freqs_real = jnp.tile(self.freqs_cis.real, (1, 2))
        freqs_imag = jnp.tile(self.freqs_cis.imag, (1, 2))

        x_rope = (x * freqs_real) + (neg_half_x * freqs_imag)
        return x_rope
```

Personally, I think one should - in the context of LLMs - decide on the max_seq_len and pad the input to that length. This way, you can avoid recompilations and make your code both more performant and easier to understand.

## How to Use RoPE in a Transformer

A last note in case you decide to use RoPE in your Transformer. As mentioned in the beginning, you need to apply RoPE after the linear layers of the queries and keys but most MHA implementations expect queries and keys with positional embeddings already applied. To use RoPE you would need to kind of "inject" the RoPE embeddings into provided MHA implementations. Because of this, you would probably need to write your own MHA implementation (or simply copy the source code of the MHA implementation of your framework of choice and modify it).

## Update

As of 2024-03-10, the Equinox library has merged the RoPE embeddings implementation into the dev branch. See the [PR](https://github.com/patrick-kidger/equinox/pull/568) for more details.
