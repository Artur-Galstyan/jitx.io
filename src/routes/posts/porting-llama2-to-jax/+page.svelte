<script lang="ts">
    import { page } from "$app/stores";
    import type { Post } from "@prisma/client";

    const post: Post = $page.data.post;
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

    <div class="flex justify-center">
        <img
            src="{$page.url.pathname}/llama2.webp"
            alt="llama2"
            class="w-1/3 mx-auto rounded-xl"
        />
    </div>

    <div class="divider" />
    <section>
        <h2>Introduction</h2>
        <p>
            In July 2023, the team at Meta have announced the
            <a
                href="https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/"
            >
                LLaMA2 LLM</a
            >, which came in 3 flavours: 7B, 13B and 70B. It was pretrained on 2
            trillion tokens and has a context length of 4096. With an RTX 3090
            (which is what I have) you'll be able to run the 7B model yourself
            (as long as you use
            <code>float16</code>). The model itself is implemented in PyTorch
            and, since I am a Jax person, I decided to port the model to Jax
            (with Equinox being the neural network library of my choice).
        </p>
        <p>
            Jax has a couple of interesting properties, not all of them are
            postitive however. What I like most about it is the functional
            programming approach that it takes. But that is a double-edged
            sword, as it means you'll have to <i>think differently</i> when writing
            Jax code, but more on that later.
        </p>
        <p>
            So, in this blog post, I'll show you my journey of porting LLaMA2 to
            Jax and talk about all the challenges along the way.
        </p>
    </section>

    <section>
        <h2>Preparations</h2>
        <p>
            If you want to follow along with this journey, you'll need a GPU
            with preferably 24 GBs of VRAM, which should give you enough
            headroom. The 7B model itself - using <code>float16</code> - only takes
            up to ~16 GBs of VRAM, but it's good to have more, as Jax is a bit more
            memory hungry as we will see in just a bit.
        </p>
    </section>
</div>
