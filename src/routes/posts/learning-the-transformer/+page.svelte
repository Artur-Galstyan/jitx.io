<script lang="ts">
    import { page } from "$app/stores";
    import HintBox from "$lib/components/HintBox.svelte";
    import type { Post } from "@prisma/client";

    const post: Post = $page.data.post;

    let index = 0;

    const i = () => {
        index += 1;
        return index;
    };

    const getStaticFile = (path: string) => {
        return `${$page.url.pathname}/${path}`;
    };
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
            encoder and the decoder.
        </p>
    </section>
    <p class="text-center text-warning">To be continued...</p>
</div>
