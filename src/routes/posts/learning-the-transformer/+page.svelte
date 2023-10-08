<script lang="ts">
    import { page } from "$app/stores";
    import HintBox from "$lib/components/HintBox.svelte";
    import Katex from "$lib/components/Katex.svelte";
    import type { Post } from "@prisma/client";
    import Figure from "$lib/components/Figure.svelte";
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

<section>
    <h2>{i()}. Introduction</h2>
    <p>
        The Transformer is one of the most successful neural network
        architectures in recent history that powers many of the new AI language
        models, such as ChatGPT, LLaMA, Claude and others. The transformer
        architecture was published in 2017 in the now famous
        <a href="https://arxiv.org/pdf/1706.03762.pdf"
            >Attention Is All You Need</a
        >
        paper and in this blog post, we'll take a deep dive into the transformer
        architecture and try to implement it in Jax (although you can follow along
        with any other NN library). In this blog post, you will learn about the crux
        of the transformer: <b>self-attention</b>. So, grab a cup of coffee,
        <b>pen and paper</b> and let's get started!
    </p>
</section>
<HintBox
    content={"You'll probably have to re-read this blog a couple of time to really understand the transformer. In a lot of places, you will see things like \"this will be explained later\", which can disrupt the flow of reading. Therefore, don't worry if you don't undestand it on the first try. Most people don't. Just keep re-reading until it clicks! 🙃"}
/>
<section>
    <h2>{i()}. Overview</h2>
    <p>
        Before we dive into the transformer, it's helpful to start with a high
        level overview first. In the following figure, you can see the
        transformer architecture.
    </p>
    <div class=" flex justify-center space-x-4">
        <figure class="flex flex-col">
            <img
                src={getStaticFile("Transformer.drawio.svg")}
                alt="Encoder"
                class="h-full"
            />
            <div class="text-sm text-gray-400 text-center">Encoder</div>
        </figure>

        <figure class="flex flex-col">
            <img
                src={getStaticFile("Decoder.drawio.svg")}
                alt="Decoder"
                class="h-full"
            />
            <div class="text-sm text-gray-400 text-center">Decoder</div>
        </figure>
    </div>
    <p>
        The traditional transformer architecture consists of two parts: the
        encoder and the decoder each of which has a certain role. Let's start
        with the encoder's role, which you can interpret as the "listener". It
        takes as input the raw tokens and generates a fixed sized vector. The
        encoder is typically added in scenarios where the input data first needs
        to be <i>encoded</i> somehow, before the output tokens are generated. For
        instance, in translation tasks (such as the original transformer paper),
        you would get as input the English text "I am happy" and your goal is to
        translate this into German. Here, you would first pass the text through the
        encoder and get as output essentially a summary of the input tokens. The
        decoder would then take that summary as input during its computation. This
        is what I meant by the encoder being the "listener".
    </p>
    <p>
        The decoder on the other hand takes the encoder's output (and we will
        see later exactly what this means) as well as its own output as input.
        This is what makes the transformer <i>auto regressive</i>: it takes its
        own output as input and generates new tokens continiously.
    </p>
    <p>
        As you can see, there are many parts in the transformer, the most
        important being the <b>Multi-Head Attention</b> block. But we're getting
        ahead of ourselves. Let's start at the beginning.
    </p>
</section>
<section>
    <h2>{i()}. The Input</h2>
    <p>
        First, we have our input tokens. In machine learning, <b>everything</b> is
        a matrix, same with our input tokens. So what are the dimensions of this
        input matrix? In general, those are
    </p>

    <Katex math={"X = [B \\times T \\times D],"} displayMode />
    <p>
        where <Katex math={"B"} /> is the batch size, <Katex math={"T"} /> is the
        number of tokens and <Katex math={"D"} /> is the
        <i>vocabulary size</i>. It can vary, depending on whatever tokeniser you
        used. If you follow along with
        <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY"
            >Andrej Karparthy's transformer tutorial</a
        >
        then <Katex math={"D"} /> is 65 (since he tokenises on the character level,
        while <code>sentencepiece</code> or <code>tiktoken</code> tokenise on the
        token-level, i.e. sub-word level). Let's have a look at some examples.
    </p>
    <pre><code class="language-python"
            >{`import tiktoken

enc = tiktoken.get_encoding("gpt2")
encoded = enc.encode("hello, world")

print(f"{encoded=}")
print(f"{enc.n_vocab=}")
`}</code
        ></pre>
    <pre><code class="language-plaintext"
            >encoded=[31373, 11, 995]
enc.n_vocab=50257
</code></pre>
    <pre><code class="language-python"
            >{`decoded = enc.decode(encoded)
print(f"{decoded=}")
`}</code
        ></pre>
    <pre><code class="language-plaintext">{`decoded='hello, world'`}</code
        ></pre>
    In this example, <Katex math={"D"} /> is 50257. But it makes sense to extend
    this number to the nearest multiple of 64 for 2 reasons: first, it will make
    <Katex math={"D"} /> divisible by 2, which will be important later and secondly,
    GPUs like multiples of 64 for efficiency reasons.
    <HintBox
        content={"Think of your 64-bit computer: if you have a 32 bit number, your computer has to perform more work, to handle a 32 bit number, while it's simplier to deal with 64 bit numbers if your system is designed with 64 bits in mind. (Extreme simplification at this point!)"}
    />
    <p>
        So, let's change the code a bit to extend <Katex math={"D"} /> to the nearest
        multiple of 64.
    </p>
    <pre><code class="language-python"
            >{`def get_next_multiple_of_64(number: int) -> int:
    while number % 64 != 0:
        number += 1
    return number

n_vocab = get_next_multiple_of_64(enc.n_vocab)
`}</code
        ></pre>
    <p>
        Next up is the input embeddings, which is a simple embedding layer. You
        embed the input (which has dimensionality <Katex math={"D"} />) into
        some - preferably smaller - dimension <Katex math={"d"} />. Let's have a
        look at the code.
    </p>

    <pre><code class="language-python"
            >{`import equinox as eqx
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
`}</code
        ></pre>
    <pre><code class="language-plaintext"
            >{`transformer.input_embedding.weight.shape=(50304, 4096)
Number of bits = 6593445888`}</code
        ></pre>
    <p>
        As you can see, our input embedding is quite large. It holds 50304 *
        4096 * 32 bits worth of information. That's around 824 megabytes, just
        for the input embedding.
    </p>

    <p>
        Next up is the positional encoding. This one is a bit tricky since there
        are many different ways to compute the positional encoding (it can even
        be learned too). However, it's way more efficient to come up with some
        strategy which adds positional information to our inputs.
    </p>
    <HintBox
        content={"This is important, because the attention mechanism is completely position agnostic. We will see a concrete example why this is the case."}
    />
    The authors used a sinusoidal positional encoding. But actually, anything goes,
    as long as you give each token a value that represents its position relative
    to all the other tokens. LLaMA, for example, uses a different kind of positional
    encoding, called "Rope Embeddings", but that's a story for another blog post.
    For now, let's implement the positional encoding:

    <pre><code class="language-python"
            >{`def get_positional_encoding(n_tokens: int, n_vocab: int) -> Float[Array, "n_tokens n_vocab"]:     
    pos = jnp.arange(n_tokens)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, n_vocab, 2) * -(jnp.log(10000.0) / n_vocab))
    # alternatively: div_term = 1 / 10000 ** (jnp.arange(0, D, 2) / D) 
    # that's closer to the actual notation they used. 
    pos_enc = jnp.zeros((n_tokens, n_vocab))
    pos_enc = pos_enc.at[:, 0::2].set(jnp.sin(pos * div_term))
    pos_enc = pos_enc.at[:, 1::2].set(jnp.cos(pos * div_term))
    return pos_enc `}</code
        ></pre>
    If you had a lot of GPUs you could even learn the positional encodings directly
    by simply adding another embedding layer to our transformer. But given how much
    memory just the input embeddings took, it might not be the most efficient idea.
    In practice, both approaches yield similar results (<span
        class="text-warning">citation needed</span
    >). And this is also precisely why we wanted to have even numbered
    dimensions earlier and why we extended <Katex math={"D"} /> to the nearest multiple
    of 64 (making it an even number). We did that, because we want to apply <Katex
        math={"sin"}
    /> and <Katex math={"cos"} /> to every second dimension, meaning we needed an
    even number of dimensions to begin with.
    <span class="text-warning"
        >This might not have been the best explanation; might improve that</span
    >

    <p>
        Our input part is almost done. All that's left is to simply compute the
        positional encodings and add them to the input matrix, before feeding
        the input to the multi-head attention (MHA) block. We will come back to
        this part when we actually start training our transformer.
    </p>
</section>
<section>
    <h2>{i()}. Multi-Head Attention</h2>
    <p>
        Okay, now we get to the most interesting part and to the heart of the
        transformer. In the earlier figures I had just shown an overview where
        MHA was just a single box, but there's a lot going on in that box. So
        let's zoom in a bit.
    </p>
    <div class="flex justify-center space-x-4">
        <figure class="flex flex-col space-y-4">
            <img
                src={getStaticFile("MHA.drawio.svg")}
                alt="MHA"
                class="h-full"
            />
            <div class="text-sm text-gray-400 text-center">
                Multi-Head Attention
            </div>
        </figure>
    </div>
    <p>
        Before we dive deep into the <i>how</i> and <i>why</i>, let's first
        identify our trainable parameters. As you can see, there are 4 linear
        layers: 3 for the <i>query, key</i> and
        <i>value</i> (whatever those are, we don't know just yet) and another one
        at the end. Let's implement a first draft and figure out the rest later.
    </p>
    <pre><code class="language-python"
            >{`class MultiHeadAttention(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear

    output: eqx.nn.Linear
    def __init__(self, ..., key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 4)

        self.query = eqx.nn.Linear(..., key=subkeys[0], use_bias=False)
        self.key = eqx.nn.Linear(..., key=subkeys[1], use_bias=False)
        self.value = eqx.nn.Linear(..., key=subkeys[2], use_bias=False)

        self.output = eqx.nn.Linear(..., key=subkeys[3], use_bias=False) 
`}</code
        ></pre>
    <p>
        We will come back and fix the above code in just a minute, but at this
        point it's time for an insertion!
    </p>
</section>
<section>
    <h2>
        Insertion: What's the difference between "normal" attention and self
        attention?
    </h2>
    <p>
        A good question! Attention in machine learning is nothing new and has
        been used plenty before. However, attention was often calculated <i
            >in relation to another query</i
        >, like a question for example. Take a look at the following example:
    </p>
    <div class="text-italic">
        <p>Query: Doughnuts are tasty! What's the capital of Germany?</p>
        <p>A: The capital of Germany is Berlin.</p>
    </div>
    <p>
        In this example, in the answer the attention of the words <i>capital</i>
        and <i>Germany</i> are high, as they coincide with the query a lot! In
        fact, if you think about it, there is little to no need to pay attention
        to the doughnut part. As you can see, in this example, attention is
        computed
        <i>relative to some other query</i>.
    </p>
    <p>
        In self attention, on the other hand, the attention scores are
        calculated <b>within the sentence itself</b>!
    </p>
    <figure class="flex flex-col space-y-4">
        <img
            src={getStaticFile("SelfAttention.drawio.svg")}
            alt="SelfAttention"
            class="h-full"
        />
        <div class="text-sm text-gray-400 text-center">
            Self Attention Example
        </div>
    </figure>
</section>
<section>
    <h2>Multi-Head Attention (again)</h2>
    <p>
        Okay, so what's up with these query, key and value vectors? The idea for
        those comes from recommender systems. Think about YouTube. What you
        search for is the <b>query</b>, for example: <i>funny cats</i>. YouTube
        has a huge database of multiple <b>keys</b> such as
        <i>video title</i>, <i>description</i> etc. Let's say there are only 3 videos
        in total on YouTube with the following titles:
    </p>
    <ul class="list list-disc">
        <li>cats walking</li>
        <li>horses running</li>
        <li>dogs jumping</li>
    </ul>
    <p>
        Your query is matched against all of these <b>keys</b> to see which one
        fits best. Let's say, their recommender systems computes these
        <b>values</b>:
    </p>
    <ul class="list list-disc">
        <li>cats walking - 0.7</li>
        <li>horses running - 0.1</li>
        <li>dogs jumping - 0.2</li>
    </ul>
    <p>
        As you can see, the first video has the highest <b>value</b>. The one
        with the highest value is the one you care most about, i.e.
        <b>pay most attention to</b>.
    </p>
    <p>
        In self attention, we don't have any external keys (otherwise we would
        have "normal" attention). In our case, the input itself is both <b
            >query</b
        >
        and <b>key</b>! This is what people mean by self attention. The query,
        key and value vector you saw in the figure above: they're all the same
        (check circle with the number 1)! The input (after it had the positional
        encodings added to it) is duplicated 3 times and each copy is named the
        query, key and value. Let's say the input is the text "How are you".
        Both query and key contain the phrase "How are you" (in the form of
        numbers though, but we will get to that). We pass those vectors through
        the linear layers and then matrix multiply those results. The goal of
        the query linear layer is to learn what parts of the query tokens each
        token should pay most attention to (2). The key linear layer learns what
        tokens are most suitable to answer to the query tokens (2). The matrix
        multiplication of these matrices yields a
        <i>score</i> matrix, which holds the self attention information. In our simplified
        example:
    </p>
    <Figure path="SASimple.drawio.svg" caption="Attention Score Matrix" />
    <p>
        The result of the matrix multiplication of <b>query</b> and
        <b>key</b> can lead to very large numbers, which is why after the multiplication
        the result is scaled down, which also allows for more stable gradients later.
        Typically, the results are scaled down relative to the chosen dimensionality.
        More specifically, the scaling is:
    </p>
    <Katex math={"\\frac{scores}{\\sqrt{d_k}},"} displayMode />
    <p>
        where <Katex math={"d_k"} /> is the dimension of the keys.
    </p>
    <HintBox
        content={"Note, how there are 3 sizes: D, which is the original input dimension, i.e. the vocabulary size. Then there's d, which is the dimension we chose for the input embedding, i.e. we embed D->d. Then there's <code>d_k</code>, which is <code> d // n_heads</code>, i.e. the embedding dimension divided <code>n_heads</code> times."}
    />
    <p>
        When we apply a softmax afterwords, we turn the attention weights into
        probabilities. At the end, we multiply the attention weights (i.e. query
        @ key) with the value vector. The idea is that the attention weights
        <i>highlight</i>
        the important parts in the value vector and <i>diminish</i> that which we
        should not pay any attention to.
    </p>
    <p>
        So far, we have only taked about a <i>single</i> attention
        <b>head</b>. In a transformer, MHAs have <i>multiple</i> heads (hence
        the name). This means the input is not split once into query, key and
        value, but rather <Katex math={"n_{heads}"} /> times (3). Oh no! Another
        parameter! <b>Quick,</b> let's add that to our implementation!
    </p>
    <pre><code class="language-python"
            >{`class MultiHeadAttention(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear

    output: eqx.nn.Linear
    def __init__(self, input_dim: int, n_heads: int, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 4)

        qkv_size = input_dim // n_heads
        
        self.query = eqx.nn.Linear(in_features=input_dim, out_features=qkv_size * n_heads, key=subkeys[0], use_bias=False)
        self.key = eqx.nn.Linear(in_features=input_dim, out_features=qkv_size * n_heads, key=subkeys[1], use_bias=False)
        self.value = eqx.nn.Linear(in_features=input_dim, out_features=qkv_size * n_heads, key=subkeys[2], use_bias=False)

        self.output = eqx.nn.Linear(in_features=input_dim, out_features=input_dim, key=subkeys[3], use_bias=False) 
`}</code
        ></pre>
    <p>
        Uff, quite a lot changed, so let's have a look. We first added the
        number of heads as a parameter to the init function. But what's up with <code
            >qkv_size</code
        >? Well, we need to split the input dimension into the number of heads.
        (Remember, that the input dimension of the MHA block is not the same as
        the original dimension, but rather whatever we chose as the embedding
        dimension for the input embeddings.) That way, each head gets the chance
        to learn something new for its dimension, independently of the other
        heads, which are all busy with their respective slice of the input
        dimension. As you can see, the output layer has the same in and out
        feature dimension (4). The reason for that is the "concatenate" block
        you saw earlier, which
        <i>concatenates</i> all the outputs of the single attention heads into a
        single matrix, which has the same shape as the original input (which was
        split <code>n_heads</code> times). So why is there a
        <code>qkv_size * n_heads</code> for each of the first 3 linear layers?
        Think of it this way: alternatively, you could have created a list of
        those linear layers and iterated each head one at a time.
        <b>But</b> that's a huge waste of computation! Instead, it's much better
        to just simply enlarge the linear layers and then - at the end - reshape
        the matrices to our desired size. We will see that in just a bit.
    </p>
    <p>
        Alright, let's implement the MHA block (without masking just yet) and
        walk through all the parts.
    </p>
    <pre><code class="language-python"
            >{`class MultiHeadAttention(eqx.Module):
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
        
        self.query = eqx.nn.Linear(in_features=input_dim, out_features=n_heads * self.qkv_size, key=subkeys[0], use_bias=False)
        self.key = eqx.nn.Linear(in_features=input_dim, out_features=n_heads * self.qkv_size, key=subkeys[1], use_bias=False)
        self.value = eqx.nn.Linear(in_features=input_dim, out_features=n_heads * self.qkv_size, key=subkeys[2], use_bias=False)

        self.output = eqx.nn.Linear(in_features=input_dim, out_features=input_dim, key=subkeys[3], use_bias=False) 

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
`}</code
        ></pre>
    <p>
        Oh, that's quite a lot. Let's break all that down, step-by-step. The
        first couple of lines are Equinox specific ways to indicate static
        fields of the module, which are not meant to be trained during
        backpropagation later. In other libraries, you'll have to check how to
        specify static fields (in PyTorch, you can just assign those to <code
            >self</code
        >).
    </p>
    <pre><code class="language-python"
            >{`# this is Equinox specific
    n_heads: int = eqx.field(static=True)
    qkv_size: int = eqx.field(static=True)
`}</code
        ></pre>
    <p>
        Then, we have the <code>__init__</code> method:
    </p>
    <pre><code class="language-python"
            >{`def __init__(self, input_dim: int, n_heads: int, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 5)

        self.qkv_size = input_dim // n_heads
        
        self.query = eqx.nn.Linear(in_features=input_dim, out_features=n_heads * self.qkv_size, key=subkeys[0], use_bias=False)
        self.key = eqx.nn.Linear(in_features=input_dim, out_features=n_heads * self.qkv_size, key=subkeys[1], use_bias=False)
        self.value = eqx.nn.Linear(in_features=input_dim, out_features=n_heads * self.qkv_size, key=subkeys[2], use_bias=False)

        self.output = eqx.nn.Linear(in_features=input_dim, out_features=input_dim, key=subkeys[3], use_bias=False) 

        self.n_heads = n_heads
`}</code
        ></pre>
    <p>
        I mentioned earlier that - since we have <Katex math={"n"} /> heads - we
        need to split the input. Currently the input has the shape of <Katex
            math={"[T \\times d]"}
        />, where <Katex math={"T"} /> is the number of tokens and <Katex
            math={"d"}
        /> is the number of dimensions of the input (which - since the input will
        come <i>after</i> the embedding dimension, will be
        <code>n_embd</code>). However, after applying the linear layers, we want
        the query, key and value matrices to have the shape <Katex
            math={"[T \\times h \\times d_k]"}
        />, where <Katex math={"h"} /> is the number of heads and <Katex
            math={"d_k"}
        /> is equal to <Katex math={"D / h"} />, i.e. the number of dimensions
        of the input divided by the number of heads. This is another reason why
        we wanted the dimension of the input embeddings earlier to be the
        nearest multiple of 64: if we make
        <code>n_heads</code>
        equal to a multiple of 32 for example, we are guaranteed an even division
        for <Katex math={"d_k"} />.
    </p>
    <p>
        This reshaping of the output into <Katex
            math={"[T \\times h \\times d_k]"}
        /> is what happens in the <code>_project</code> method:
    </p>
    <pre><code class="language-python"
            >{`def _project(self, proj, x):
        seq_length, _ = x.shape # seq_len is T
        projection = jax.vmap(proj)(x) # vmap over T, since x has the shape T x D
        return projection.reshape(seq_length, self.n_heads, -1) # reshape into T x h x d
`}</code
        ></pre>
    <p>To be sure you can check with this:</p>
    <pre><code class="language-python"
            >{`assert q.shape == (T, self.n_heads, self.qkv_size)
assert k.shape == (T, self.n_heads, self.qkv_size)
assert v.shape == (T, self.n_heads, self.qkv_size)
`}</code
        ></pre>
    <HintBox
        content={"Transformers nowadays have evolved and can allow for different query, key and value sizes. But that's a topic for another blog post."}
    />
    <HintBox
        content={"And just to reiterate: <code>d_k</code> is the number of input dimensions divided by the number of heads. <code>D</code> is the original number of dimensions and <code>d</code> is the number of dimensions we wanted to embed the original <code>D</code> dimensions into (i.e. D->d)."}
    />
    <p>
        Now that we have the query, key and value, it's time to compute the
        attention scores:
    </p>
    <pre><code class="language-python"
            >{`dot_product_vmap = jax.vmap(
    lambda q, k: jnp.dot(q, k.T), 
    in_axes=(1, 1), 
    out_axes=1
)
attention_scores = dot_product_vmap(q, k)
`}</code
        ></pre>
    <p>
        In the lambda function, we pass the query and key as parameter.
        Remember, that their shapes is <Katex
            math={"T \\times h \\times d_k"}
        />. We want the attention scores to have the shape of <Katex
            math={"T \\times h \\times T"}
        />. If you're wondering why, have a look at the following figure.
    </p>

    <Figure path={"ThT.drawio.svg"} caption="Attention Matrix" />
    <p>
        And that's why we pass as <code>in_axes = (1, 1)</code> because that
        corresponds to the <Katex math={"h"} /> dimension in <Katex
            math={"T\\times h\\times d_k"}
        />. We specified <code>out_axis = 1</code> because we want to "place"
        the dimension we <code>vmap</code> over into axis 1, which means that
        the output matrix will have the <Katex math={"h"} /> dimension on axis 1,
        which is exactly what we want, since we want the attention scores to have
        the shape of <Katex math={"T \\times h \\times T"} />. Next up, we apply
        the scaling and softmax, which is quite simple:
    </p>
    <pre><code class="language-python"
            >{`attention_scores = attention_scores / jnp.sqrt(self.qkv_size)
attention_scores = jax.nn.softmax(attention_scores)`}</code
        ></pre>
    <p>
        Now, we need to multiply our scaled, softmaxed scores with the value
        matrix from the beginning.
    </p>
    <pre><code class="language-python"
            >{`matmul_vmap = jax.vmap(
    lambda s, v: jnp.dot(s, v), 
    in_axes=(1, 1), 
    out_axes=1
)

output = matmul_vmap(attention_scores, v)
`}</code
        ></pre>
    <p>
        Let's clarify our shapes again: the attention scores have the shape <Katex
            math={"T \\times h \\times T"}
        /> and <code>v</code> has the shape of <Katex
            math={"T \\times h \\times d_k"}
        />. At the end, we want to pass the output through the
        <code>output</code>
        linear layer, which expects the shape to be <Katex
            math={"T \\times d"}
        />, where <Katex math={"d"} /> is the number of dimensions of the input (which
        in turn is the number of output dimensions of the input embeddings). As you
        can see, the "batch dimension" that we want to
        <code>vmap</code>
        over and which is equal for both the scores and <code>v</code> is the
        axis 1, which is <Katex math={"h"} />. To be even more specific, this is
        what happens:
    </p>
    <ol class="list list-disc">
        <li>
            We dot product <Katex math={"T \\times h \\times T"} /> with <Katex
                math={"T \\times h \\times d_k"}
            />
        </li>
        <li>
            By specifying the <code>in_axis=(1, 1)</code>, we tell Jax
            <code>vmap</code> over axis 1; so for now let's plug that axis out (we
            will place it back in later)
        </li>
        <li>
            Now, we're left with these: <Katex
                math={"T \\times T \\cdot T \\times d_k"}
            /> and axis <Katex math={"h"} /> waiting on the sidelines.
        </li>
        <li>
            The resulting matrix has the shape <Katex math={"T \\times d_k"} />
        </li>
        <li>
            If we don't specify the <code>out_axis</code> explicitly, then Jax
            would place it to axis 0, which would give us the result <Katex
                math={"h \\times T \\times d_k"}
            />. But, we want to plug it back in into axis 1, which would make
            the output matrix have the shape <Katex
                math={"T \\times h \\times d_k"}
            /> instead, which is what we want.
        </li>
    </ol>
    <p>
        Reshaping the vector to <Katex math={"T, -1"} /> makes the first dimension
        be <Katex math={"T"} /> and the -1 specifies to multiply the remaining shapes.
        In our case, the multiplication is <Katex math={"h \\cdot d_k"} /> and if
        you remember <Katex math={"d_k = d / h"} />, then you'll notice that we
        end up with our original input dimensionality <Katex
            math={"h \\cdot d_k = D"}
        />:
    </p>
    <pre><code class="language-python">{`output = output.reshape(T, -1)`}</code
        ></pre>
    <p>
        Lastly, passing the output through the output layer, gives us a final
        output matrix with the shape <Katex math={"T \\times d"} />
        And that's it. This was the MHA block for the encoder
        <b>without masking</b>.
    </p>
    <p>
        Let's generalise our implementation a bit and add the masking part. But
        first we need some more background about masking and why we need it.
    </p>
</section>
<section>
    <h2>Insertion: Masking</h2>
    <p>
        In general, the decoder takes the output of the encoder (in an
        encoder-decoder architecture) and generates data off of it while also
        using its own previous output as additional input. This makes the
        decoder <i>autoregressive</i>, i.e. it gets its own output as input.
        During training, the encoder has access to the entire input sequence and
        therefore needs no masking (as it's not autoregressive). The decoder on
        the other hand <b>does</b> need masking. Let's take translation as the
        task at hand here, where we have an encoder and a decoder. The goal is
        to translate from English to German. The encoder gets as input sequence
        "I am happy", processes it and generates a <i>summary vector</i>, which
        the decoder gets as input. The decoder's goal is to translate that into
        German <i>Ich bin glücklich</i>. Let's say the input vector for the
        decoder is <Katex math={"[SV, START, ICH]"} />, simplified to <Katex
            math={"[SV, S, I]"}
        />. The current goal is to predict the next word "bin" given the summary
        vector, the start token and the word "ich". Since the input is <Katex
            math={"[SV, START, ICH]"}
        />, we currently don't even need masking yet, as the goal of masking
        <b
            >is to prevent the decoder to have access to future tokens, which
            are the targets itself</b
        >. The decoder computes the self-attention for the word "ich" given all
        the previous tokens:
    </p>
    <Katex
        displayMode
        math={"att('ICH')= \\alpha SV + \\beta S + \\gamma I,"}
    />
    <p>
        where the greeks are just some weight parameters. In this example the
        weight <Katex math={"\\gamma"} />should be high, since that is very
        important to predict the next word "bin". However, if the training data
        was instead <Katex math={"[SV, START, ICH, BIN]"} />, then, when the
        decoder computes the attention for "ich", it would be
    </p>
    <Katex
        displayMode
        math={"att('ICH')= \\alpha SV + \\beta S + \\gamma I + \\eta B"}
    />
    <p>
        The decoder's goal is to predict "bin", so it shouldn't know about the
        word "bin". Yet, it computes the attention with <Katex
            math={"\\eta \\cdot BIN"}
        />, i.e. the very word it's trying to predict and learn. This is not
        correct. Therefore, we need to hide the word "bin" from the decoder.
        Consider the mask vector <Katex math={"[1, 1, 1, 0]"} />. If we multiply
        the mask vector with our weight vector <Katex
            math={"[\\alpha, \\beta, \\gamma, \\eta]"}
        />, then the attention of the word "bin" becomes:
    </p>
    <Katex
        displayMode
        math={"att('ICH')= (1 \\cdot \\alpha) SV + (1 \\cdot\\beta) S + (1 \\cdot\\gamma) I + (0 \\cdot\\eta) B,"}
    />
    <p>
        which effectively zeroes out all future, unwanted tokens. Let's now
        implement masking in our MHA block.
    </p>
</section>
<section>
    <h2>Multi-Head Attention (again, again)</h2>
    <p>
        I alluded earlier to the special mask vector and now, I'll show you how
        to generate it. Let's look at a smaller example, just so we can
        visualise everything:
    </p>
    <pre><code class="language-python"
            >{`T = 4
tril = jnp.tril(jnp.ones(shape=(T, T)))
print(tril)
mask = jnp.where(tril == 0, jnp.full(shape=(T, T), fill_value=float("-inf")), jnp.zeros(shape=(T,T)))
print(f"{mask}")
mask = jax.nn.softmax(mask, axis=-1)
print(f"{mask}")
`}</code
        ></pre>
    <pre><code class="language-text"
            >{`
[[1. 0. 0. 0.]
 [1. 1. 0. 0.]
 [1. 1. 1. 0.]
 [1. 1. 1. 1.]]
[[  0. -inf -inf -inf]
 [  0.   0. -inf -inf]
 [  0.   0.   0. -inf]
 [  0.   0.   0.   0.]]
[[1.         0.         0.         0.        ]
 [0.5        0.5        0.         0.        ]
 [0.33333334 0.33333334 0.33333334 0.        ]
 [0.25       0.25       0.25       0.25      ]]
`}</code
        ></pre>
    <p>
        In the code, <code>tril</code> is a triangular matrix, with zeroes above
        its diagonal. With <code>jnp.where</code>, we can generate a <Katex
            math={"T \\times T"}
        /> matrix and change the values to negative infinity, where the
        <code>tril</code>
        matrix had zeroes, i.e. <code>tril == 0</code>. When we softmax that, we
        get a nice probability distribution, that sums to 1.
    </p>
    <p>Here's another implementation:</p>
    <pre><code class="language-python"
            >{`T = 4
mask = jnp.tril(jnp.ones(shape=(T, T))) == 1
print(mask)
logits = jax.random.uniform(shape=(T, T), key=jax.random.PRNGKey(0))
logits = jnp.where(mask, logits, float("-inf"))
print(logits)
logits = jax.nn.softmax(logits, axis=-1)
print(logits)
`}</code
        ></pre>
    <pre><code class="language-text"
            >{`
[[ True False False False]
 [ True  True False False]
 [ True  True  True False]
 [ True  True  True  True]]
[[0.5338           -inf       -inf       -inf]
 [0.6309322  0.20438278       -inf       -inf]
 [0.21696508 0.32493377 0.7355863        -inf]
 [0.3715024  0.1306243  0.04838264 0.60753703]]
[[1.         0.         0.         0.        ]
 [0.6050494  0.39495057 0.         0.        ]
 [0.26359332 0.29364634 0.44276032 0.        ]
 [0.26482752 0.20813785 0.19170524 0.3353294 ]]
`}</code
        ></pre>
    <p>
        In this implementation, we first generate the mask, but this time we
        make it a boolean mask directly. Then, we say, that the logits are equal
        to the logits, where the mask is <code>true</code> otherwise it's negative
        infinity. When we softmax that, we get - once again - a nice probability
        distribution.
    </p>
    <p>
        Let's add the second implementation to our code. In fact, I'll just show
        the entire finished MHA implementation:
    </p>
    <pre><code class="language-python"
            >{`
class MultiHeadAttention(eqx.Module):
    n_heads: int = eqx.field(static=True)
    qkv_size: int = eqx.field(static=True)

    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear

    output: eqx.nn.Linear
    def __init__(self, input_dim: int, dim: int, n_heads: int, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 5)

        self.qkv_size = input_dim // n_heads
        
        self.query = eqx.nn.Linear(in_features=input_dim, out_features=n_heads * self.qkv_size, key=subkeys[0], use_bias=False)
        self.key = eqx.nn.Linear(in_features=input_dim, out_features=n_heads * self.qkv_size, key=subkeys[1], use_bias=False)
        self.value = eqx.nn.Linear(in_features=input_dim, out_features=n_heads * self.qkv_size, key=subkeys[2], use_bias=False)

        self.output = eqx.nn.Linear(in_features=input_dim, out_features=input_dim, key=subkeys[3], use_bias=False) 

        self.n_heads = n_heads

    def _project(self, proj, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.n_heads, -1)

    def __call__(self, x: Array, masking: bool):
        T, _ = x.shape

        q = self._project(self.query, x)
        k = self._project(self.key, x)
        v = self._project(self.value, x)

        assert q.shape == (T, self.n_heads, self.qkv_size)
        assert k.shape == (T, self.n_heads, self.qkv_size)
        assert v.shape == (T, self.n_heads, self.qkv_size)

        dot_product_vmap = jax.vmap(
            lambda q, k: jnp.dot(q, k.T), 
            in_axes=(1, 1), 
            out_axes=1
        )
        attention_scores = dot_product_vmap(q, k)
        print(f"{attention_scores.shape=}")
        attention_scores = attention_scores / jnp.sqrt(self.qkv_size)
        if masking:
            mask = jnp.tril(jnp.ones(shape=(T, T))) == 1
            mask = jnp.expand_dims(mask, axis=1) # we add an extra dimension at axis 1 for broadcasting
            attention_scores = jnp.where(mask, attention_scores, float("-inf"))
            print(f"{attention_scores}")
        attention_scores = jax.nn.softmax(attention_scores, axis=-1)
        print(f"{attention_scores}")
        matmul_vmap = jax.vmap(
            lambda s, v: jnp.dot(s, v), 
            in_axes=(1, 1), 
            out_axes=1
        )

        output = matmul_vmap(attention_scores, v)
        print(f"before reshaping {output.shape=}")
        output = output.reshape(T, -1)
        print(f"after reshaping {output.shape=}")
        output = jax.vmap(self.output)(output)

        return output
    
n_vocab = 128
N_HEADS = 2
T = 4 # 4 tokens 
mha = MultiHeadAttention(
    input_dim=n_vocab,
    dim=N_EMBD,
    n_heads=N_HEADS,
    key=jax.random.PRNGKey(21)
)

x = jax.random.uniform(shape=(T, n_vocab), key=jax.random.PRNGKey(11))
output = mha(x, True)
`}</code
        ></pre>
    <p>
        And that's it. That's the MHA block with optional masking. Great! Let's
        add that to our transformer class. You probably forgot about that,
        right? No worries. Here's the current transformer implementation:
    </p>
    <pre><code class="language-python"
            >{`class Transformer(eqx.Module):
    input_embedding: eqx.nn.Embedding

    masked_mha: MultiHeadAttention
    

    def __init__(self, n_dims: int, n_embd: int, n_heads: int, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 20) # let's just split 20 for now, we'll probably need them later
        self.input_embedding = eqx.nn.Embedding(n_dims, n_embd, key=subkeys[0])
        dim = n_dims // n_heads
        self.masked_mha = MultiHeadAttention(input_dim=n_dims, dim=dim, n_heads=n_heads, key=subkeys[1])

key = jax.random.PRNGKey(42)
INPUT_DIMS=65
N_EMBD = 4096
N_HEADS = 4
transformer = Transformer(n_dims=n_vocab, n_embd=N_EMBD, n_heads=N_HEADS, key=key)
`}</code
        ></pre>
</section>
<section>
    <h2>{i()}. The Rest</h2>
    <p>
        We came quite far and to be honest, most of the hard work is already
        done with the multi-head attention block. But if we look at the
        transformer architecture again, we notice a couple of missing things,
        which are - luckily - very easy to implement. Here it is again for
        reference:
    </p>

    <figure class="flex flex-col">
        <img
            src={getStaticFile("Decoder.drawio.svg")}
            alt="Decoder"
            class="h-full w-1/2 mx-auto"
        />
        <div class="text-sm text-gray-400 text-center">Decoder</div>
    </figure>
    <p>
        <b
            >For the remainder of this blog post, we will focus on the decoder
            only.</b
        >
        The other trainable part is the feedforward neural network. That's something
        anyone can do in their sleep, so let's write it out:
    </p>
    <pre><code class="language-python"
            >{`class Transformer(eqx.Module):
    input_embedding: eqx.nn.Embedding
    masked_mha: MultiHeadAttention
    feedforward: eqx.nn.MLP

    def __init__(self, n_dims: int, n_embd: int, n_heads: int, key: PRNGKeyArray, width_size: int=32, depth: int = 2) -> None:
        key, *subkeys = jax.random.split(key, 20) # let's just split 20 for now, we'll probably need them later
        self.input_embedding = eqx.nn.Embedding(n_dims, n_embd, key=subkeys[0])
        dim = n_dims // n_heads
        self.masked_mha = MultiHeadAttention(input_dim=n_dims, dim=dim, n_heads=n_heads, key=subkeys[1])

        # Equinox has a built-in MLP module
        self.feedforward = eqx.nn.MLP(in_size=n_dims, out_size=n_dims, width_size=width_size, key=subkeys[2], depth=depth)
`}</code
        ></pre>
    <p>
        What's missing now are these <code>add & norm</code> parts. The add part
        is important, because after applying the MHA to the input, the input is
        distorted quite a lot. By adding it back, we effectively remind the
        transformer of how initial structure of the input. As your
        normalisation: it is required, because our network is quite big and big
        networks need normalisation in order to train effectively. Here's the
        RMSNorm implementation which <strike>stole</strike> borrowed from the LLaMA2
        implementation:
    </p>
    <pre><code class="language-python"
            >{`
class RMSNorm(eqx.Module):
    weight: Array
    eps: float

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = jnp.ones(dim)

    def _norm(self, x: Array):
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: Array) -> Array:
        output = self._norm(x)
        return output * self.weight

`}</code
        ></pre>
    <p>
        Let me now show you the full implementation of the (simple) transformer:
    </p>
    <pre><code class="language-python"
            >{`
class Transformer(eqx.Module):
    input_embedding: eqx.nn.Embedding
    masked_mha: MultiHeadAttention
    feedforward: eqx.nn.MLP
    rms_norm: RMSNorm

    output: eqx.nn.Linear
    positional_encoding: Array 

    def __init__(self, n_dims: int, n_embd: int, n_heads: int, key: PRNGKeyArray, width_size: int=32, depth: int = 2, max_token_size: int = 8) -> None:
        key, *subkeys = jax.random.split(key, 20) # let's just split 20 for now, we'll probably need them later
        self.input_embedding = eqx.nn.Embedding(n_dims, n_embd, key=subkeys[0])
        self.masked_mha = MultiHeadAttention(input_dim=n_embd, n_heads=n_heads, key=subkeys[1])

        # Equinox has a built-in MLP module
        self.feedforward = eqx.nn.MLP(in_size=n_embd, out_size=n_embd, width_size=width_size, key=subkeys[2], depth=depth)
        self.positional_encoding = get_positional_encoding(max_token_size, n_embd)

        self.rms_norm = RMSNorm(dim=n_embd)

        self.output = eqx.nn.Linear(in_features=n_embd, out_features=n_dims, key=subkeys[4], use_bias=False)

    def __call__(self, x):
        print(f"side effect")
        x = self.input_embedding(x)
        x += self.positional_encoding
        x = self.rms_norm(self.masked_mha(x, masking=True) + x) # residual connection
        x = self.rms_norm(jax.vmap(self.feedforward)(x) + x) # residual connection
        x = jax.vmap(self.output)(x)
        # x = jax.nn.softmax(x) # we don't softmax here, because we want the raw logits for our loss function 
        # but you can totally softmax here and inverse that later; 
        return x 
`}</code
        ></pre>
</section>
<section>
    <h2>{i()}. Training and Evaluation</h2>
    <p>
        Our transformer is done. It lacks a couple of improvements, such as
        dropout for example. But that's fine. This is more for learning purposes
        anyway. The following shows a training loop, which uses the
        TinyShakespeare dataloader, which loads the same tiny Shakespeare
        dataset that Andrej used in his tutorial. I won't go into detail though,
        as this is just "regular" engineering and is not specific to the
        transformer.
    </p>

    <pre><code class="language-python"
            >{`
from tinyshakespeareloader.hamlet import get_data


data = get_data()


train_dataloader, test_dataloader, vocabulary_size, chars, encode, decode = data["train_dataloader"], data["test_dataloader"], data["vocabulary_size"], data["chars"], data["encode"], data["decode"]
key = jax.random.PRNGKey(420)
INPUT_DIMS: int = int(vocabulary_size)
N_EMBD = 32
N_HEADS = 4
MAX_T = 8

def loss_fn(transformer: Transformer, x: Array, y: Array):
    logits = eqx.filter_vmap(transformer)(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)

    return jnp.mean(loss)

def evaluate(transformer: Transformer, test_dataloader):
    loss = 0
    jitted_loss_fn = eqx.filter_jit(loss_fn)
    for x, y in test_dataloader:
        x = jnp.array(x.numpy())
        y = jnp.array(y.numpy())
        loss += jitted_loss_fn(transformer, x, y)
    
    return loss / len(test_dataloader)

@eqx.filter_jit
def step(transformer: PyTree, opt_state: optax.OptState, optimiser: optax.GradientTransformation, x: Array, y: Array):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(transformer, x, y)
    updates, opt_state = optimiser.update(grads, opt_state, transformer)
    transformer = eqx.apply_updates(transformer, updates)
    return transformer, opt_state, loss

transformer = Transformer(n_dims=INPUT_DIMS, n_embd=N_EMBD, n_heads=N_HEADS, key=key)
#start_loss = evaluate(transformer, test_dataloader)
#print(f"{start_loss=}")
optimiser = optax.adamw(learning_rate=0.001)
opt_state = optimiser.init(eqx.filter(transformer, eqx.is_inexact_array))
for i, (x, y) in enumerate(train_dataloader):
    x = jnp.array(x.numpy())
    y = jnp.array(y.numpy())
    transformer, opt_state, loss = step(transformer, opt_state, optimiser, x, y)
    if i % 100 == 0:
        eval_loss = evaluate(transformer, test_dataloader)
        print(f"{i}. {loss=}, {eval_loss=}")

print("done.")
print(f"{evaluate(transformer, test_dataloader)=}")
`}</code
        ></pre>
    <p>
        If you run this - as is - it won't really be that good. You'll have to
        scale it up by a lot to see <b>real</b> improvements. Furthermore, we
        only have a <b>single</b> MHA block.
        <b>In practice, you would create a sequence of these MHA blocks.</b>
        Like a multi-layer feedforward network, except where each layer is a MHA
        block. That is left to the reader as an exercise.
    </p>
</section>
<p>
    And that's it! That was the transformer. I hope you liked it and will join
    me again, on my next engineering blog post. Thanks for reading and I will
    see you in the next one 👋
</p>
