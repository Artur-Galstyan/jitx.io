<script lang="ts">
    import { page } from "$app/stores";
    import HintBox from "$lib/components/HintBox.svelte";
    import Katex from "$lib/components/Katex.svelte";
    import type { Post } from "@prisma/client";
    import { onMount } from "svelte";

    import Figure from "$lib/components/Figure.svelte";
    import Icon from "svelte-icons-pack";
    import AiOutlineGithub from "svelte-icons-pack/ai/AiOutlineGithub";

    const post: Post = $page.data.post;

    let index = 0;

    const i = () => {
        index += 1;
        return index;
    };
    let idxJ = 0;
    const j = () => {
        idxJ += 1;
        return idxJ;
    };

    const getStaticFile = (path: string) => {
        return `${$page.url.pathname}/${path}`;
    };
</script>

<svelte:head>
    <title>
        {post.title}
    </title>
</svelte:head>

<div class="text-justify mx-auto">
    <div class="divider" />
    <section>
        <p>
            Multihead-Attention (MHA) is the key part of the transformer
            architecture. It uses a mechanism called self attention, which has
            been very successful in NLP tasks. I already introduced the
            transformer in my other <a href="/posts/learning-the-transformer">
                blog post</a
            >. In this blog post, we will take a deepdive into the MHA and try
            to make it as general as we can. Let's start with an overview of the
            MHA block. In the following figure, you can see the structure of the
            MHA block.
        </p>
        <Figure path={"MHA.drawio.svg"} caption="MHA Block" />
        <p>
            The input to the MHA block is duplicated into 3 vectors: the query,
            key and value vectors. Each of those is first passed through their
            respective linear layer. Each of those layers has a specific task:
        </p>
        <ul>
            <li>
                Query Layer: transforms the input to query vectors, i.e. what
                you're interested in
            </li>
            <li>
                Key Layer: transforms the input to a set of keys to match the
                query vectors against
            </li>
            <li>
                Value Layer: take the scaled combination of the query and key
                projects and compute the output of the MHA block
            </li>
        </ul>

        <p>This is a good starting point, so let's write this down in code.</p>
        <HintBox
            content={'(By the way, most of this was already covered in my <a class="link" href="/posts/learning-the-transformer"> previous blog post </a> and this implementation takes heavy inspiration from already existing implementations such as the <a class="link" href="https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/_attention.py">MHA block from Equinox</a>)'}
        />
        <p>
            One thing to note is the dimensionalities of the vectors, so let's
            start by defining the dimensions first. Here's the notation for this
            blog post:
        </p>
        <ul>
            <li>
                <Katex math={"L"} />: maximum sequence length
            </li>
            <li>
                <Katex math={"h"} />: number of heads
            </li>
            <li>
                <Katex math={"\\{q,k,v\\}_{emdb}"} />: query, key or value
                embedding dimension
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
            Usually, the query, key and value input dimensions are the same, but
            we want our implementation to be very general and make as few
            assumptions as possible about the current use case. Therefore, we
            will be more specific. The reason that normally they are the same is
            that, typically, they come out of the input embeddings (with the
            positional embeddings added on top) and the same embedding is used
            for all vectors, giving them all the same input dimension. Let's
            start with some imports and define the dimensions.
        </p>
    </section>
    <section>
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
            As shown in the MHA figure, we will first implement the linear layer
            parts.
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
            As mentioned in my previous blog post, a MHA block consists of
            multiple heads. But, instead of looping over each head, one at a
            time, we can instead simply enlarge the query, key and value layers
            to include all of the heads. Look at it this way: taking all of the
            heads into consideration, the output shape of, say, the query
            projection should be:
            <Katex math={"[L \\times h \\times q_{embd}]"} displayMode />

            By making the query projection layer project from <Katex
                math={"q_{in}"}
            /> to <Katex math={"h * q_{emdb}"} />, we get initially a matrix
            with the same <Katex math={"[L \\times h * q_{embd}]"} />. From
            there, we can simply reshape that matrix into our desired shape: <Katex
                math={"[L \\times h \\times q_{embd}]"}
            />. This is just the first steps for the query and key projections;
            they have still quite the journey ahead. We still need to matrix
            multiply, scale (sometimes mask) and softmax them. Let's write a
            function that can do all of that in one go.
        </p>
    </section>
</div>
