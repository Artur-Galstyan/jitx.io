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

<div class="prose text-justify">
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

        <div class="text-sm text-gray-400 text-center">An image of a LLaMA</div>
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
    <section>
        <h2>{i()}. Overview</h2>
        <p>
            Before we dive into the transformer, it's helpful to start with a
            high level overview first. In the following figure, you can see the
            transformer architecture.
        </p>
        <div class="flex justify-center space-x-4">
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
        >).

        <p>
            Our input part is almost done. All that's left is to simply compute
            the positional encodings and add them to the input matrix, before
            feeding the input to the multi-head attention (MHA) block. We will
            come back to this part when we actually start training our
            transformer.
        </p>
        <section>
            <h2>{i()}. Multi-Head Attention</h2>
            <p>
                Okay, now we get to the most interesting part and to the heart
                of the transformer.
            </p>
        </section>
    </section>
    <p class="text-center text-warning">To be continued...</p>
</div>
