<script lang="ts">
    import HintBox from "$lib/components/HintBox.svelte";
    import Katex from "$lib/components/Katex.svelte";

    import Figure from "$lib/components/Figure.svelte";
</script>

<p>
    Multihead-Attention (MHA) is the key part of the transformer architecture.
    It uses a mechanism called self attention, which has been very successful in
    NLP tasks. I already introduced the transformer in my other <a
        href="/posts/learning-the-transformer"
    >
        blog post</a
    >. In this blog post, we will take a deepdive into the MHA and try to make
    it as general as we can. Let's start with an overview of the MHA block. In
    the following figure, you can see the structure of the MHA block.
</p>
<Figure path={"MHA.drawio.svg"} caption="MHA Block" />
<p>
    The input to the MHA block is duplicated into 3 vectors: the query, key and
    value vectors. Each of those is first passed through their respective linear
    layer. Each of those layers has a specific task:
</p>
<ul>
    <li>
        Query Layer: transforms the input to query vectors, i.e. what you're
        interested in
    </li>
    <li>
        Key Layer: transforms the input to a set of keys to match the query
        vectors against
    </li>
    <li>
        Value Layer: take the scaled combination of the query and key projects
        and compute the output of the MHA block
    </li>
</ul>

<p>This is a good starting point, so let's write this down in code.</p>
<HintBox
    content={'By the way, most of this was already covered in my <a class="link" href="/posts/learning-the-transformer"> previous blog post </a> and this implementation takes heavy inspiration from already existing implementations, especially from the <a class="link" href="https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_attention.py">MHA block from Equinox</a>. In fact, by the end of this blog post, we will arrive at the present implementation from the MHA used in Equinox with some useful extras (hopefully)'}
/>
<p>
    One thing to note is the dimensionalities of the vectors, so let's start by
    defining the dimensions first. Here's the notation for this blog post:
</p>
<ul>
    <li>
        <Katex math={"L"} />: maximum sequence length
    </li>
    <li>
        <Katex math={"h"} />: number of heads
    </li>
    <li>
        <Katex math={"\\{q,k,v\\}_{emdb}"} />: query, key or value embedding
        dimension
    </li>
</ul>
<p>Furthermore, let's define the input to the MHA block.</p>
<ul>
    <li>
        Query: <Katex math={"[L \\times q_{in}]"} />, where <Katex
            math={"q_{in}"}
        /> is the query input dimension
    </li>
    <li>
        Key: <Katex math={"[L \\times k_{in}]"} />, where <Katex
            math={"k_{in}"}
        /> is the key input dimension
    </li>
    <li>
        Value <Katex math={"[L \\times v_{in}]"} />, where <Katex
            math={"v_{in}"}
        /> is the value input dimension
    </li>
</ul>
<p>
    Usually, the query, key and value input dimensions are the same, but we want
    our implementation to be very general and make as few assumptions as
    possible about the current use case. Therefore, we will be more specific.
    The reason that normally they are the same is that, typically, they come out
    of the input embeddings (with the positional embeddings added on top) and
    the same embedding is used for all vectors, giving them all the same input
    dimension. Let's start with some imports and define the dimensions.
</p>

<pre><code class="language-python"
        >{`
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
`}</code
    ></pre>
<p>
    As shown in the MHA figure, we will first implement the linear layer parts.
</p>
<pre><code class="language-python"
        >{`
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
`}</code
    ></pre>
<pre><code class="language-text"
        >{`
query.shape=(10, 4, 32) # L x h x q_embd
key.shape=(10, 4, 32) # L x h x k_emdb
value.shape=(10, 4, 32) # L x h x v_emdb
            `}</code
    ></pre>
<p>
    As mentioned in my previous blog post, a MHA block consists of multiple
    heads. But, instead of looping over each head, one at a time, we can instead
    simply enlarge the query, key and value layers to include all of the heads.
    Look at it this way: taking all of the heads into consideration, the output
    shape of, say, the query projection should be:
    <Katex math={"[L \\times h \\times q_{embd}]"} displayMode />

    By making the query projection layer project from <Katex math={"q_{in}"} /> to
    <Katex math={"h * q_{emdb}"} />, we get initially a matrix with the same <Katex
        math={"[L \\times h * q_{embd}]"}
    />. From there, we can simply reshape that matrix into our desired shape: <Katex
        math={"[L \\times h \\times q_{embd}]"}
    />. This is just the first steps for the query and key projections; they
    have still quite the journey ahead. We still need to matrix multiply, scale
    (sometimes mask) and softmax them. Let's write a function that can do all of
    that in one go.
</p>
<pre><code class="language-python"
        >{`
def dot_product_attention(query_projection: Array, key_projection: Array, value_projection: Array, mask: Optional[Array | None]) -> Array:
    attention_weights = jax.vmap(
        lambda q, k: q @ k.T,
        in_axes=(1, 1),
        out_axes=1
    )(query_projection, key_projection)
    attention_weights = attention_weights / jnp.sqrt(key_projection.shape[-1])
    attention_weights = jax.nn.softmax(attention_weights, axis=-1)

    return attention_weights
`}</code
    ></pre>
<p>
    Masking is missing from the above implementation for now, but that's ok. We
    will add that in later.
</p>
<p>
    Let's think for a moment what we even did in the function above and what our
    goal even was. First, we took the query and the key projections, i.e. the
    matrices that came <b>after</b> plugging our initial input through the linear
    layers. The task of the linear layers is to learn how to transform the input
    into usable queries and keys, such that they can be matched against each other.
</p>
<p>
    Let's just imagine for a moment that our linear layers do a <b
        >perfect job</b
    > of transforming the input into the perfect queries and keys. What do we even
    do with those? This is where the self-attention mechanism comes into play.
</p>
<HintBox
    content={`If you've read my <a class="link" href=\"/posts/learning-the-transformer"
    >previous post</a
> you can skip the following explanation.`}
/>
<p>
    To understand <i>self-attention</i>, we first need to understand
    <i>"regular" attention</i>. Here's a useful example. Think of YouTube and
    how you might want to find a particular video. You enter a
    <b>query</b>
    into the search bar. Now, how does YouTube know, which video to show you? Each
    video has a video title, a description and some other metadata. Those are the
    <b>keys</b>. Your query is matched against the keys and whichever has the
    highest <b>value</b> (according to whatever metric was using in the
    <i>matching-process</i>), is shown to you first.
</p>
<Figure path="Attention1.drawio.svg" caption="Regular Attention" />
<p>
    In that example, the query and keys are different. The query was what you
    searched for and the keys whatever YouTube has in its database. In
    self-attention on the other hand, the query and key are the same - at least
    initially.
</p>
<p>
    Take the input sentence: "Hi, how are". That sentence is both the query and
    the key, albeit in its <i>raw form</i>. Matching those against each other
    directly isn't really useful - not to mention that it's always the same. We
    need to turn these raw query and keys into <b>usable</b>
    queries and keys, such that you can actually match one against the other.
    <i>How</i>
    we can do that specifically is unknown. We can, however,
    <b>learn it</b>. And that's precisely the goal of the linear layers.
</p>
<Figure path="Attention2.drawio.svg" caption="Self-Attention" />
<p>
    Alright, let's continue with the MHA block, while keeping in mind, that we
    will add masking <b>last</b>. In our image we see that the next step is the
    matrix multiplication of the scaled attention weights with the value matrix.
</p>
<pre><code class="language-python"
        >{`
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
`}</code
    ></pre>
<p>
    As you can see, we compute the matrix multiplication between the attention
    weights and the value. We use <code>jax.vmap</code> to map over axis 1 of the
    inputs. Just as a reminder, the shape of the attention weights and the values
    are:
</p>
<Katex math={"L \\times h \\times L"} displayMode />
<Katex math={"L \\times h \\times v_{embd},"} displayMode />
<p>
    where the first is the shape of the attention weight matrix and the latter
    is the shape of the value projection and <Katex math={"h"} /> is the number of
    heads. Why is the shape of the attention weights <Katex
        math={"L \\times h \\times L"}
    />? We mentioned earlier how in self-attention the query and the key are
    both the input (that went through the linear layers, but from now on,
    whenever I reference the query and keys you can assume that I'm referring to
    the ones <b>after</b> having their linear layers applied to them). When we
    compute the how much attention each word gives to another word in the
    sequence, then it only makes sense that we have a square matrix! If your
    query has 3 words, then you have 3 keys; matching each query with each key
    gives you a 3x3 matrix. From there, we have to remind ourselves, that we
    have a <i>multihead attention</i> block, which means that each head -
    <b>individually and independently</b> - computes the attention weights. Since
    we want to use matrices to leverage efficient GPU usage, we keep everything in
    a tensor and simply add the head dimension in the middle. It's in the middle
    simply for consistency, as we chose the middle dimension (i.e. axis 1) to be
    the number of heads.
</p>
<p>
    By defining <code>jax.vmap(..., in_axes=(1, 1), out_axes=1)</code> we tell
    Jax to vectorise across axes (1, 1) for the input (which is <Katex
        math={"h"}
    />) and then we define <code>out_axes=1</code>, which puts that axes back
    into the output in axis position 1. The following figure illustrates this.
</p>
<div class="w-3/4 mx-auto">
    <Figure path="VMAP-MHA.drawio.svg" caption="VMAP Example" />
</div>
<p>
    Now, we need to concatenate the heads and then apply the final linear output
    layer. We're still missing the output layer in our MHA implementation, so
    first let's define the output's dimension:
</p>
<pre><code class="language-python"
        >{`
output_dim = 32
`}</code
    ></pre>
<p>
    And let's add the output to our current implementation, both in the <code
        >__init__</code
    >
    and the <code>__call__</code> functions.
</p>
<pre><code class="language-python"
        >{`
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
`}</code
    ></pre>
<pre><code class="language-text"
        >{`
ic| query.shape: (10, 4, 32)

ic| key.shape: (10, 4, 32)
ic| value.shape: (10, 4, 32)
ic| attention_weights.shape: (10, 4, 10)
ic| qkv_matmul.shape: (10, 4, 32)
ic| concatenation.shape: (10, 128)
ic| output.shape: (10, 32)
`}</code
    ></pre>
<HintBox
    content={`I'm using the <a class="link" href="https://github.com/gruns/icecream">icecream package</a> for logging, which I've found to be very useful.`}
/>
<p>
    As you can see, the output shape is <Katex math={"L \\times q_{embd}"} />,
    but it needn't necessarily be; it's just usually the case.
</p>
<p>
    Our implementation is already ok-ish flexible, but we can go one step
    further, namely by looking at the <b>number of heads</b>! So far, we just
    assumed that each head should have 1 query, key and value, but this need not
    be the case! In fact, there is something called
    <a class="link" href="https://arxiv.org/abs/1911.02150"
        >Multi-Query Attention (MQA)</a
    >, where <b>one</b> key and value matrix is shared across all heads.
    Personally, I would have gone with the name
    <i>Single-Key-Value Attention (SKVA)</i> just to highlight that it's the keys
    and the values that change.
</p>
<p>
    Let's review. Our current implementation (the standard MHA block) has <Katex
        math={"h"}
    /> heads and each head has a query, key and value. This means in MHA, we have
    <Katex math={"h"} /> queries, keys and values. In SKVA (i.e. MQA) we have <Katex
        math={"h"}
    /> heads and <Katex math={"h"} /> queries but only 1 key and value. So, let's
    go one step further and define
    <i>Multi-Key-Value Attention (MKVA)</i>, where we will have <Katex
        math={"h"}
    /> heads, <Katex math={"h"} /> queries and
    <Katex math={"d"} /> keys and values.
</p>
<HintBox
    content={`In this case, MHA would have <code>d = h</code> and in SKVA (MQA) we would have <code>d = 1</code>.`}
/>
<p>Here's the difference between SKVA (MQA) and MHA:</p>
<div class="flex space-x-4">
    <Figure path="StandardMHA.drawio.svg" caption="Standard MHA" />
    <div class="divider divider-horizontal" />
    <Figure path="SKVA.drawio.svg" caption="SKVA (MQA)" />
</div>
<p>
    As you can see, <code>key 1</code> and <code>value 1</code> are shared across
    all heads. Now, let's think for a moment what MKVA would look like. Here's a
    first draft:
</p>
<Figure path="MKVA.drawio.svg" caption="MKVA First Draft" />
<HintBox
    content={"We're treading into untapped waters here. Implementations from the big players such as PyTorch or TensorFlow aren't as customisable, which means I couldn't find a reference implementation."}
/>
<p>
    Let's start simple and give these dimensions a name: <code
        >key_multihead_dim</code
    >, <code>value_multihead_dim</code> and <code>query_multihead_dim</code>,
    although I'm not sure if we will even need last one at all. Also, it might
    very well be that what I'm trying to do here is not even possible, but let's
    try it anyway. First, let's add these dimensions to the
    <code>__init__</code> method.
</p>
<pre><code class="language-python"
        >{`
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
`}</code
    ></pre>

<p>
    We need to come up with a solid plan of attack for our generic
    implementation; most importantly, we need to figure out <i
        >what we actually want to achieve</i
    >, which is harder to articulate than you might think. Basically, if you set
    e.g. <code>key_multihead_dim</code> to any number lower than the number of
    heads, it effectively means that not all heads will have a key. For example,
    if your MHA block has 8 heads and only a single shared key, i.e.
    <code>key_multihead_dim = 1</code>, then all heads "share" that one key
    (SKVA/MQA). But if you set it to, say, <code>key_multihead_dim=7</code>,
    then 7 heads have a key and one does not. But in matrix multiplication you
    can't simply "skip" a dimension, which is going to be our main challenge.
</p>

<p>
    The current plan of attack is to use a double <code>vmap</code> strategy,
    which I call the <i>outer</i> and <i>inner</i> <code>vmap</code>. The
    following figure shows a rough outline of my strategy.
</p>
<Figure
    width={600}
    height={600}
    path="MKVANew.drawio.svg"
    caption="Outline of the Double VMAP Strategy"
/>
<p>
    The rough idea is to have kind-of a nested for-loop. The outer <code
        >vmap</code
    >
    maps over axis 1 of the keys and values while the inner <code>vmap</code>
    maps over axis 1 of the queries (which is usually the number of heads). Upon
    closer inspection it would probably make more sense to have a single value for
    the <code>multihead_dim</code>, since - at the very least - the key and
    value dimensions must be the same (otherwise, we'd need a third
    <code>vmap</code>
    for
    <code>value_multihead_dim</code>). Actually, let's do that right now.
</p>
<pre><code class="language-python"
        >{`...
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
`}</code
    ></pre>
<p>
    After performing the MKVA, the out_axis is placed (per default) on <code
        >out_axes=0</code
    >
    and we'd perform a summation over that <code>out_axes</code> and then
    compute the mean over the <code>out_axes</code>.
</p>
<p>
    Doing it this way, is actually <b>not</b> 100% mathematically equivalent to
    what we had before! As it turns out, these <code>vmap</code>s are not
    exactly the same:
</p>
<pre><code class="language-python"
        >{`
vmap(fn, in_axes=(1, 1, 1))
vmap(vmap(fn, in_axes=(None, 1, 1)), in_axes=(1, None, None))
`}</code
    ></pre>
<p>
    For now, we will keep it and later train a transformer with our MHA
    implementation to experimentally check if our new approach even works at
    all.
</p>
<p>
    Let's implement our new strategy. First, we will update the <code
        >dot_product_attention</code
    >
    function to compute the <code>qkv_matmul</code> matrix directly using the value
    projections, while also keeping in mind our new matrix dimensions, which I have
    added to the function signature:
</p>
<pre><code class="language-python"
        >{`def dot_product_attention(
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
`}</code
    ></pre>
<p>
    Here is the part that will perform the inner <code>vmap</code>
</p>
<pre><code class="language-python"
        >{`
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
`}</code
    ></pre>

<p>
    Let's also adjust our MHA implementation to use the double <code>vmap</code>
    strategy:
</p>
<pre><code class="language-python"
        >{`
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
`}</code
    ></pre>
<pre><code class="language-text"
        >{`
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
`}</code
    ></pre>

<p>As you can see, the output shape matches exactly what we had before.</p>
<p>
    The next step is to add masking, which we can simply copy from the <a
        class="link"
        href="multihead-attention">other blog post</a
    >. If we build a transformer around this MHA implementation and test our
    implementation on the TinyShakespeare dataset, then (at least in my
    preliminary tests) it doesn't seem to work! I'm not exactly sure why, but I
    suspect that it has something to do with the <code>vmap</code> strategy. I will
    investigate this further in the future and publish my findings in another post.
</p>
