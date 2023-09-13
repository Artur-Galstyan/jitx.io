<script lang="ts">
    import { page } from "$app/stores";
    import HintBox from "$lib/components/HintBox.svelte";
    import Katex from "$lib/components/Katex.svelte";
    import type { Post } from "@prisma/client";
    import { onMount } from "svelte";
    import hljs from "highlight.js";
    import python from "highlight.js/lib/languages/python";
    hljs.registerLanguage("python", python);
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
    onMount(() => {
        hljs.highlightAll();
    });
</script>

<div class="prose text-justify mx-auto">
    <h1 class="text-center font-extrabold py-0 my-0">
        {post.title}
    </h1>
    <h4 class="text-center text-gray-500 text-sm my-0 py-0">
        {new Date(post.createdAt).toLocaleDateString("en-CA", {
            year: "numeric",
            month: "2-digit",
            day: "2-digit"
        })}
    </h4>

    <div class="flex justify-center flex-col mx-auto">
        <img
            src="{$page.url.pathname}/llama2.webp"
            alt="llama2"
            class="w-1/3 mx-auto rounded-xl p-0 m-0 mt-4"
        />

        <div class="text-sm text-gray-400 text-center">An Image of a LLaMA</div>
    </div>

    <div class="divider" />
    <section>
        <h2>{i()}. Introduction</h2>
        <p>
            The Transformer is one of the most successful neural network
            architectures in recent history that powers many of the new AI
            language models, such as ChatGPT, LLaMA, Claude and others. The
            transformer architecture was published in 2017 in the now famous
            <a href="https://arxiv.org/pdf/1706.03762.pdf"
                >Attention Is All You Need</a
            >
            paper and in this blog post, we'll take a deep dive into the transformer
            architecture and try to implement it in Jax (although you can follow
            along with any other NN library). In this blog post, you will learn about
            the crux of the transformer: <b>self-attention</b>. So, grab a cup
            of coffee, <b>pen and paper</b> and let's get started!
        </p>
    </section>
    <HintBox
        content={"You'll probably have to re-read this blog a couple of time to really understand the transformer. In a lot of places, you will see things like \"this will be explained later\", which can disrupt the flow of reading. Therefore, don't worry if you don't undestand it on the first try. Most people don't. Just keep re-reading until it clicks! 🙃"}
    />
    <section>
        <h2>{i()}. Overview</h2>
        <p>
            Before we dive into the transformer, it's helpful to start with a
            high level overview first. In the following figure, you can see the
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
            encoder and the decoder each of which has a certain role. Let's
            start with the encoder's role, which you can interpret as the
            "listener". It takes as input the raw tokens and generates a fixed
            sized vector. The encoder is typically added in scenarios where the
            input data first needs to be <i>encoded</i> somehow, before the output
            tokens are generated. For instance, in translation tasks (such as the
            original transformer paper), you would get as input the English text
            "I am happy" and your goal is to translate this into German. Here, you
            would first pass the text through the encoder and get as output essentially
            a summary of the input tokens. The decoder would then take that summary
            as input during its computation. This is what I meant by the encoder
            being the "listener".
        </p>
        <p>
            The decoder on the other hand takes the encoder's output (and we
            will see later exactly what this means) as well as its own output as
            input. This is what makes the transformer <i>auto regressive</i>: it
            takes its own output as input and generates new tokens continiously.
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
            First, we have our input tokens. In machine learning, <b
                >everything</b
            > is a matrix, same with our input tokens. So what are the dimensions
            of this input matrix? In general, those are
        </p>

        <Katex math={"X = [B \\times T \\times D],"} displayMode />
        <p>
            where <Katex math={"B"} /> is the batch size, <Katex math={"T"} /> is
            the number of tokens and <Katex math={"D"} /> is the number of
            <i>embedding dimensions</i>. Those can vary, depending on whatever
            tokeniser you used. If you follow along with
            <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY"
                >Andrej Karparthy's transformer tutorial</a
            >
            then <Katex math={"D"} /> is 65 (since he tokenises on the character
            level, while <code>sentencepiece</code> or <code>tiktoken</code> tokenise
            on the token-level, i.e. sub-word level). Let's have a look at some examples.
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
            So, let's change the code a bit to extend <Katex math={"D"} /> to the
            nearest multiple of 64.
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
            Next up is the input embeddings, which is a simple embedding layer.
            You embed the input (which has dimensionality <Katex math={"D"} />)
            into some - preferably smaller - dimension <Katex math={"d"} />.
            Let's have a look at the code.
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
            4096 * 32 bits worth of information. That's around 824 megabytes,
            just for the input embedding.
        </p>

        <p>
            Next up is the positional encoding. This one is a bit tricky since
            there are many different ways to compute the positional encoding (it
            can even be learned too). However, it's way more efficient to come
            up with some strategy which adds positional information to our
            inputs.
        </p>
        <HintBox
            content={"This is important, because the attention mechanism is completely position agnostic. We will see a concrete example why this is the case."}
        />
        The authors used a sinusoidal positional encoding. But actually, anything
        goes, as long as you give each token a value that represents its position
        relative to all the other tokens. LLaMA, for example, uses a different kind
        of positional encoding, called "Rope Embeddings", but that's a story for
        another blog post. For now, let's implement the positional encoding:

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
        by simply adding another embedding layer to our transformer. But given how
        much memory just the input embeddings took, it might not be the most efficient
        idea. In practice, both approaches yield similar results (<span
            class="text-warning">citation needed</span
        >). And this is also precisely why we wanted to have even numbered
        dimensions earlier and why we extended <Katex math={"D"} /> to the nearest
        multiple of 64 (making it an even number). We did that, because we want to
        apply <Katex math={"sin"} /> and <Katex math={"cos"} /> to every second dimension,
        meaning we needed an even number of dimensions to begin with.
        <span class="text-warning"
            >This might not have been the best explanation; might improve that</span
        >

        <p>
            Our input part is almost done. All that's left is to simply compute
            the positional encodings and add them to the input matrix, before
            feeding the input to the multi-head attention (MHA) block. We will
            come back to this part when we actually start training our
            transformer.
        </p>
    </section>
    <section>
        <h2>{i()}. Multi-Head Attention</h2>
        <p>
            Okay, now we get to the most interesting part and to the heart of
            the transformer. In the earlier figures I had just shown an overview
            where MHA was just a single box, but there's a lot going on in that
            box. So let's zoom in a bit.
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
            identify our trainable parameters. As you can see, there are 4
            linear layers: 3 for the <i>query, key</i> and
            <i>value</i> (whatever those are, we don't know just yet) and another
            one at the end. Let's implement a first draft and figure out the rest
            later.
        </p>
        <pre><code class="language-python"
                >{`class MultiHeadAttention(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear

    output: eqx.nn.Linear
    def __init__(self, qkv_size: int, dim: int, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 4)

        # These dimensions are wrong! We will fix them shortly though :)
        self.query = eqx.nn.Linear(in_features=qkv_size, out_features=dim, key=subkeys[0], use_bias=False)
        self.key = eqx.nn.Linear(in_features=qkv_size, out_features=dim, key=subkeys[1], use_bias=False)
        self.value = eqx.nn.Linear(in_features=qkv_size, out_features=dim, key=subkeys[2], use_bias=False)

        self.output = eqx.nn.Linear() # I'll leave this one open for now, but this will be discussed shortly, I promise ;)
`}</code
            ></pre>
        <p>
            We will come back and fix the above code in just a minute, but at
            this point it's time for an insertion!
        </p>
    </section>
    <section>
        <h2>
            Insertion: What's the difference between "normal" attention and self
            attention?
        </h2>
        <p>
            A good question! Attention in machine learning is nothing new and
            has been used plenty before. However, attention was often calculated <i
                >in relation to another query</i
            >, like a question for example. Take a look at the following
            example:
        </p>
        <div class="text-italic">
            <p>Query: Doughnuts are tasty! What's the capital of Germany?</p>
            <p>A: The capital of Germany is Berlin.</p>
        </div>
        <p>
            In this example, in the answer the attention of the words <i
                >capital</i
            >
            and <i>Germany</i> are high, as they coincide with the query a lot!
            In fact, if you think about it, there is little to no need to pay
            attention to the doughnut part. As you can see, in this example,
            attention is computed
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
            Okay, so what's up with these query, key and value vectors? The idea
            for those comes from recommender systems. Think about YouTube. What
            you search for is the <b>query</b>, for example: <i>funny cats</i>.
            YouTube has a huge database of multiple <b>keys</b> such as
            <i>video title</i>, <i>description</i> etc. Let's say there are only
            3 videos in total on YouTube with the following titles:
        </p>
        <ul>
            <li>cats walking</li>
            <li>horses running</li>
            <li>dogs jumping</li>
        </ul>
        <p>
            Your query is matched against all of these <b>keys</b> to see which
            one fits best. Let's say, their recommender systems computes these
            <b>values</b>:
        </p>
        <ul>
            <li>cats walking - 0.7</li>
            <li>horses running - 0.1</li>
            <li>dogs jumping - 0.2</li>
        </ul>
        <p>
            As you can see, the first video has the highest <b>value</b>. The
            one with the highest value is the one you care most about, i.e.
            <b>pay most attention to</b>.
        </p>
        <p>
            In self attention, we don't have any external keys (otherwise we
            would have "normal" attention). In our case, the input itself is
            both <b>query</b> and <b>key</b>! This is what people mean by self
            attention. The query, key and value vector you saw in the figure
            above: they're all the same (check circle with the number 1)! The
            input (after it had the positional encodings added to it) is
            duplicated 3 times and each copy is named the query, key and value.
            Let's say the input is the text "How are you". Both query and key
            contain the phrase "How are you" (in the form of numbers though, but
            we will get to that). We pass those vectors through the linear
            layers and then matrix multiply those results. The goal of the query
            linear layer is to learn what parts of the query tokens each token
            should pay most attention to (2). The key linear layer learns what
            tokens are most suitable to answer to the query tokens (2). The
            matrix multiplication of these matrices yields a
            <i>score</i> matrix, which holds the self attention information. In our
            simplified example:
        </p>
        <figure class="flex flex-col space-y-4">
            <img
                src={getStaticFile("SASimple.drawio.svg")}
                alt="Example"
                class="h-full"
            />
            <div class="text-sm text-gray-400 text-center">
                Attention Score Matrix
            </div>
        </figure>
        <p>
            The result of the matrix multiplication of <b>query</b> and
            <b>key</b> can lead to very large numbers, which is why after the
            multiplication the result is scaled down, which also allows for more
            stable gradients later. Typically, the results are scaled down
            relative to the chosen dimensionality (what we named
            <code>dim</code> in our code above). More specifically, the scaling is:
        </p>
        <Katex math={"\\frac{scores}{\\sqrt{d_k}},"} displayMode />
        <p>
            where <Katex math={"d_k"} /> is our chosen dimension for the linear layers.
            When we apply a softmax afterwords, we turn the attention weights into
            probabilities. At the end, we multiply the attention weights (i.e. query
            @ key) with the value vector. The idea is that the attention weights
            <i>highlight</i>
            the important parts in the value vector and <i>diminish</i> that which
            we should not pay any attention to.
        </p>
        <p>
            So far, we have only taked about a <i>single</i> attention
            <b>head</b>. In a transformer, MHAs have <i>multiple</i> heads
            (hence the name). This means the input is not split once into query,
            key and value, but rather <Katex math={"n_{heads}"} /> times (3). Oh
            no! Another parameter! <b>Quick,</b> let's add that to our implementation!
        </p>
        <pre><code class="language-python"
                >{`class MultiHeadAttention(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear

    output: eqx.nn.Linear
    def __init__(self, input_dim: int, dim: int, n_heads: int, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 4)

        qkv_size = input_dim // n_heads
        
        self.query = eqx.nn.Linear(in_features=input_dim, out_features=dim * n_heads, key=subkeys[0], use_bias=False)
        self.key = eqx.nn.Linear(in_features=input_dim, out_features=dim * n_heads, key=subkeys[1], use_bias=False)
        self.value = eqx.nn.Linear(in_features=input_dim, out_features=dim * n_heads, key=subkeys[2], use_bias=False)

        self.output = eqx.nn.Linear(in_features=input_dim, out_features=input_dim, key=subkeys[3], use_bias=False) 
`}</code
            ></pre>
        <p>
            Uff, quite a lot changed, so let's have a look. We first added the
            number of heads as a parameter to the init function. But what's up
            with <code>qkv_size</code>? Well, we need to split the input
            dimension <Katex math={"D"} /> into the number of heads. That way, each
            head gets the chance to learn something new for its dimension, independently
            of the other heads, which are all busy with their respective slice of
            the input dimension. As you can see, the output layer has the same in
            and out feature dimension (4). The reason for that is the "concatenate"
            block you saw earlier, which
            <i>concatenates</i> all the outputs of the single attention heads
            into a single matrix, which has the same shape as the original input
            (which was split <code>n_heads</code> times). So why is there a
            <code>dim * n_heads</code> for each of the first 3 linear layers?
            Think of it this way: alternatively, you could have created a list
            of those linear layers and iterated each head one at a time.
            <b>But</b> that's a huge waste of computation! Instead, it's much better
            to just simply enlarge the linear layers and then - at the end - reshape
            the matrices to our desired size. We will see that in just a bit.
        </p>
        <p>
            Alright, let's implement the MHA block (without masking just yet)
            and walk through all the parts.
        </p>
        <pre><code class="language-python"
                >{`class MultiHeadAttention(eqx.Module):
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
            backpropagation later. In other libraries, you'll have to check how
            to specify static fields (in PyTorch, you can just assign those to <code
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
                >{`def __init__(self, input_dim: int, dim: int, n_heads: int, key: PRNGKeyArray) -> None:
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
            I mentioned earlier that - since we have <Katex math={"n"} /> heads -
            we need to split the input. Currently the input has the shape of <Katex
                math={"[T \\times D]"}
            />, where <Katex math={"T"} /> is the number of tokens and <Katex
                math={"D"}
            /> is the number of dimensions of the input (see
            <code>enc.n_vocab</code>). However, after applying the linear
            layers, we want the query, key and value matrices to have the shape <Katex
                math={"[T \\times h \\times d]"}
            />, where <Katex math={"h"} /> is the number of heads and <Katex
                math={"d"}
            /> is equal to <Katex math={"D / h"} />, i.e. the number of
            dimensions of the original input divided by the number of heads.
            This is another reason why we wanted the dimension to be the nearest
            multiple of 64: if we make <code>n_heads</code> equal to a multiple
            of 32 for example, we are guaranteed an even division for <Katex
                math={"d"}
            />.
        </p>
        <p />
    </section>
    <p class="text-center text-warning">To be continued...</p>
</div>
