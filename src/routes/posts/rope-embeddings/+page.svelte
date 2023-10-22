<script lang="ts">
    import Figure from "$lib/components/Figure.svelte";
    import HintBox from "$lib/components/HintBox.svelte";
    import Katex from "$lib/components/Katex.svelte";
</script>

<section>
    <h3>Introduction</h3>
    <p>
        Self-attention in a transformer is position-invariant, meaning that it
        the attention matrix has the same values regardless of the order of the
        words. Here's an example and notice how the values (which are made up by
        the way) are the same just at different positions.
    </p>
    <Figure
        path="PosInvar.drawio.svg"
        caption="Position-Invariant Attention Matrix"
    />
    <p>
        This is a problem because the order of the words in a sentence is
        important. For example, "I like cats" and "cats like I" have the same
        words but different meanings and we wish to capture that. This is why,
        in the original Transformer paper, the authors added positional
        embeddings to the input, namely the sinusoidal embeddings. Here's a code
        snippet, which calculates those positional embeddings:
    </p>
    <pre><code class="language-python"
            >{`
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
    `}</code
        ></pre>
    <p>
        Once you have those positional encodings, you add them to the input. The
        core idea is to generate some unique position vector for each word in
        the sentence and add it to the word embedding. This way, even if the
        words are the same, the position vectors will be different and thus
        attention matrix will be different. In the case of sinusoidal
        embeddings, you can visualize it as generating lots of sine & cosine
        waves with different frequencies and stack those up. Then, for each
        position, you draw a that passes through all sine waves and measure the
        value for each wave. That process would look like this:
    </p>
    <Figure path="Sinus1.drawio.svg" caption="Sinusoidal Embeddings" />
    <p>
        While it does give each word a unique position, it doesn't do it in a
        predictable way nor does it keep the relative distances between the
        words constant. We can easily visualise that by looking at the word
        vectors as we apply the sinusoidal embeddings:
    </p>
    <Figure path="sinusanimation.avifs" caption="Sinusoidal Embeddings" />
    <p>
        As you can see, the arrow jumps around and the relative distances
        change. It's hard to predict where the arrow will jump to next.
    </p>
</section>
<section>
    <h3>RoPE Embeddings</h3>
    <p>
        This is where we learn about RoPE embeddings. RoPE stands for Rotary
        Positional Embeddings and was published in the <a
            href="https://arxiv.org/pdf/2104.09864.pdf">RoFormer paper</a
        >. The core idea of RoPE embeddings is to use a rotation matrix to
        rotate the input vector. This way, the relative distances between the
        words are preserved and the position vectors are predictable. Take a
        look at the vector animation again, but this time with RoPE embeddings.
        Notice, how the arrow moves in a predictable way.
    </p>
    <Figure path="ropeanimation.avifs" caption="RoPE Embeddings" />
    <p>
        Much better, right? In this blog post, we will implement a <code
            >RoPEEmbeddingsLayer</code
        > in Equinox and hopefully get that merged into the main repository.
    </p>
</section>
<section>
    <h3>The Math</h3>
    <p>
        The core idea is very simple. Once you have your query and key vectors
        (the ones you get from passing your input through the linear layers),
        you rotate them some angle <Katex math={"m\\theta"} />, where <Katex
            math={"m"}
        /> is the token position and <Katex math={"\\theta"} /> is the rotation angle.
        For example, if you have 3 tokens, then <Katex math={"m"} /> would be 0,
        1, and 2. To rotate a vector, you multiply it by the rotation matrix:
    </p>
    <Katex
        math={`
\\begin{aligned}
    \\begin{bmatrix}
        \\cos\\theta & -\\sin\\theta \\\\
        \\sin\\theta & \\cos\\theta
    \\end{bmatrix}
    \\begin{bmatrix}
        x \\\\
        y
    \\end{bmatrix}
    &=
    \\begin{bmatrix}
        x\\cos\\theta - y\\sin\\theta \\\\
        x\\sin\\theta + y\\cos\\theta
    \\end{bmatrix}
\\end{aligned}
    `}
        displayMode
    />
    <HintBox
        content={`I'm not 100% sure why we apply the rotations <b>after</b> applying the linear weights. My guess is that if we applied them <b>before</b> the 
    linear layers, then we would just distort the input vectors and the rotation effect would be lost. I'm not sure though. So if you know, let me know in the comments.`}
    />
    <p>To calculate the rotation angle, we use the following formula:</p>
    <Katex
        math={`
\\theta_i = 10000^{-\\frac{2(i - 1)}{d}} 
    ,`}
        displayMode
    />
    <p>
        where <Katex math={"d"} /> is the dimensionality of the tokens (in the example
        above it was 2 but in practice it's a much bigger number). To extend this
        to the multidimensional case where we have more than just 2 dimensions, we
        would simply construct a sparse rotation matrix for each dimension and split
        the input vector into chunks of 2 and rotate each chunk separately. But that's
        a super slow operation and in practice we use this formula:
    </p>
    <Katex
        math={`
    R^{d}_{\\theta, m} \\mathbf{x} = \\begin{pmatrix}
x_1 \\\\
x_2 \\\\
x_3 \\\\
x_4 \\\\
\\vdots \\\\
x_{d-1} \\\\
x_d \\\\
\\end{pmatrix} \\otimes \\begin{pmatrix}
\\cos(m\\theta_1) \\\\
\\cos(m\\theta_1) \\\\
\\cos(m\\theta_2) \\\\
\\cos(m\\theta_2) \\\\
\\vdots \\\\
\\cos(m\\theta_{d/2}) \\\\
\\cos(m\\theta_{d/2}) \\\\
\\end{pmatrix} + \\begin{pmatrix}
-x_2 \\\\
x_1 \\\\
-x_4 \\\\
x_3 \\\\
\\vdots \\\\
-x_{d-1} \\\\
x_d \\\\
\\end{pmatrix} \\otimes \\begin{pmatrix}
\\sin(m\\theta_1) \\\\
\\sin(m\\theta_1) \\\\
\\sin(m\\theta_2) \\\\
\\sin(m\\theta_2) \\\\
\\vdots \\\\
\\sin(m\\theta_{d/2}) \\\\
\\sin(m\\theta_{d/2}) \\\\
\\end{pmatrix}

        
    `}
        displayMode
    />
    <p>
        So, let's implement this! There are 2 ways to implement this: the
        real-numbers way and the complex-numbers way. Let's start with the
        former.
    </p>
</section>
<section>
    <h3>The Real-Numbers Way</h3>
    <p>
        The <Katex math={"cos"} /> and <Katex math={"sin"} /> parts can be precomputed
        and cached since they don't change at all. As mentioned before, the formula
        for the rotation angle is:
    </p>
    <Katex
        math={`
\\theta_i = 10000^{-\\frac{2(i - 1)}{d}}
    `}
        displayMode
    />
    <pre><code class="language-python"
            >{`

def get_freqs(seq_len: int, dim: int, base: int = 10000):
    theta = 1 / (base ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(seq_len)

    idx_theta = jnp.einsum("i,j->ij", t, theta)
    idx_theta = jnp.concatenate([idx_theta, idx_theta], axis=1)

    freqs_cos = jnp.cos(idx_theta)
    freqs_sin = jnp.sin(idx_theta)

    return freqs_cos, freqs_sin 
    `}</code
        ></pre>
    <p>
        The code above simply calculates all the indices first (i.e. <Katex
            math={"m\\theta"}
        />) and then simply applies <Katex math={"cos"} /> and <Katex
            math={"sin"}
        /> to each of those. The next step is to negate every other element in the
        input vector <Katex math={"x"} /> before applying the frequencies.
    </p>
    <pre><code class="language-python"
            >{`
def get_rope_embedding(x: Float[Array, "seq_len dim"]):
    def negate_half(x: Float[Array, "seq_len dim"]):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([x[..., :d_2], -x[..., d_2:]], axis=-1)

    seq_len, dim = x.shape
    cos, sin = get_freqs(seq_len, dim)
    neg_half_x = negate_half(x)
    x_rope = (x * cos) + (neg_half_x * sin)
    return x_rope
`}</code
        ></pre>
    <p>That's really all there is to it.</p>
</section>
<section>
    <h3>The Complex-Numbers Way</h3>
    <p>
        This is a bit more complicated but it's also faster if you JIT the code.
        Notice, in the function above, we return <code
            >freqs_cos, freqs_sin</code
        > as 2 separate arrays. As it turns out, we can express both of them as complex
        numbers:
    </p>
    <pre><code class="language-python"
            >{`
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
    `}</code
        ></pre>
    <p>
        In the code above, we store the <Katex math={"cos"} /> values in the real
        part and the <Katex math={"sin"} /> values in the imaginary part. With the
        precomputed frequencies, we can now implement the RoPE embeddings:
    </p>
    <pre><code class="language-python"
            >{`
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
`}</code
        ></pre>
    <p>
        We can verify that both implementations yield the same result with a
        simple test:
    </p>
    <pre><code class="language-python"
            >{`
import jax.numpy as jnp
from icecream import ic


dim = 1024
max_seq_len = 4096

x_rope_real = get_rope_embedding(jnp.ones((max_seq_len, dim)))
x_rope_complex = get_rope_embeddings_complex(jnp.ones((max_seq_len, dim)))

ic(x_rope_real)
ic(x_rope_complex)

ic(jnp.allclose(x_rope_real, x_rope_complex))
`}</code
        ></pre>
    <pre><code class="language-text"
            >{`
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
`}</code
        ></pre>
</section>
<section>
    <h3>Performance Comparison</h3>
    <p>
        Now, you might be wondering which implementation is faster. Let's find
        out by benchmarking them:
    </p>
    <pre><code class="language-python"
            >{`
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
`}</code
        ></pre>
    <p>When we run this, we get the following results:</p>
    <table class="table">
        <thead>
            <tr>
                <td />
                <th>Complex</th>
                <th>Real</th>
            </tr>
        </thead>
        <tr>
            <td>No JIT</td>
            <td>299.1253</td>
            <td>193.1669</td>
        </tr>
        <tr>
            <td>Avg No Jit</td>
            <td>0.029912</td>
            <td>0.01931</td>
        </tr>
        <tr>
            <td>With JIT</td>
            <td>71.4320</td>
            <td>83.8433</td>
        </tr>
        <tr>
            <td>Avg With Jit</td>
            <td>0.0071</td>
            <td>0.0083</td>
        </tr>
    </table>
    <p>
        Interestingly, the real-numbers way is faster than the complex-numbers
        way when we don't JIT the code. However, once we JIT the code, the
        complex-numbers way is faster. We will go with the second implementation
        given the pretense that - when using JAX - you should <b>always</b> strive
        to JIT your code.
    </p>
</section>
<section>
    <h3>Equinox Implementation</h3>
    <p>
        Let's write the RoPE embeddings for the Equinox library. Our
        implementation will have the constraint that the input's sequence length
        must remain the same, which makes our code less dynamic. The reason for
        this is that JAX - in general - does <b>not</b> like dynamic shapes at all,
        since when the shapes change, functions need to be recompiled. Here's the
        implementation:
    </p>
    <pre><code class="language-python"
            >{`
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

`}</code
        ></pre>
    <p>
        Personally, I think one should - in the context of LLMs - decide on the <code
            >max_seq_len</code
        > and pad the input to that length. This way, you can avoid recompilations
        and make your code both more performant and easier to understand.
    </p>
</section>
<section>
    <h3>How to Use RoPE in a Transformer</h3>
    <p>
        A last note in case you decide to use RoPE in your Transformer. As
        mentioned in the beginning, you need to apply RoPE <b>after</b> the
        linear layers of the queries and keys but most MHA implementations
        expect queries and keys with positional embeddings already applied. To
        use RoPE you would need to kind of
        <i>"inject"</i> the RoPE embeddings into provided MHA implementations. Because
        of this, you would probably need to write your own MHA implementation (or
        simply copy the source code of the MHA implementation of your framework of
        choice and modify it).
    </p>
</section>
