<script lang="ts">
    import python from "highlight.js/lib/languages/python";
    import c from "highlight.js/lib/languages/c";
    import hljs from "highlight.js";
    import { onMount } from "svelte";
    import type { Post } from "@prisma/client";
    import { page } from "$app/stores";
    import Icon from "svelte-icons-pack";
    import AiOutlineGithub from "svelte-icons-pack/ai/AiOutlineGithub";
    import CommentSection from "$lib/components/CommentSection.svelte";
    import AiFillUpCircle from "svelte-icons-pack/ai/AiFillUpCircle";
    import { fade } from "svelte/transition";

    hljs.registerLanguage("python", python);
    hljs.registerLanguage("c", c);
    const post: Post = $page.data.post;

    onMount(() => {
        hljs.highlightAll();
    });

    const getStaticFile = (path: string) => {
        return `${$page.url.pathname}/${path}`;
    };

    let scrollY = 0;
</script>

<svelte:window bind:scrollY />
<svelte:head>
    <title>
        {post.title}
    </title>
    <meta name="description" content={post.shortDescription} />
    <meta name="keywords" content={post.tags.join(", ")} />
    <meta name="author" content="Artur A. Galstyan" />
    <meta name="robots" content="index, follow" />
    <meta property="og:title" content={post.title} />
    <meta property="og:description" content={post.shortDescription} />
    <meta property="og:image" content={getStaticFile("thumbnail.webp")} />
    <meta property="og:url" content={$page.url.pathname} />

    <!-- Facebook Meta Tags -->
    <meta property="og:url" content={$page.url.pathname} />
    <meta property="og:type" content="website" />
    <meta property="og:title" content="One Week of C" />
    <meta property="og:description" content={post.shortDescription} />
    <meta property="og:image" content={getStaticFile("thumbnail.webp")} />

    <!-- Twitter Meta Tags -->
    <meta name="twitter:card" content="summary_large_image" />
    <meta property="twitter:domain" content="jitx.io" />
    <meta property="twitter:url" content={$page.url.pathname} />
    <meta name="twitter:title" content="One Week of C" />
    <meta name="twitter:description" content={post.shortDescription} />
    <meta name="twitter:image" content={getStaticFile("thumbnail.webp")} />
</svelte:head>
<article class="flex justify-center relative">
    <div class="w-[95%] md:w-[50%] mx-auto">
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

        {#if post.thumbnail}
            <div class="flex justify-center flex-col mx-auto">
                <img
                    src="{$page.url.pathname}/thumbnail.webp"
                    alt="thumbnail"
                    class="w-1/3 mx-auto rounded-xl p-0 m-0 mt-4"
                />

                <div class="text-sm text-gray-400 text-center">
                    {post.thumbnailDescription}
                </div>
            </div>
        {/if}
        {#if post.repositoryLink}
            <div class="flex justify-center my-12 mask mask-circle">
                <button
                    on:click={() => {
                        if (post.repositoryLink) {
                            window.location.href = post.repositoryLink;
                        }
                    }}
                >
                    <Icon
                        src={AiOutlineGithub}
                        size="64"
                        color="purple"
                        className="transition hover:bg-green-100 ease-in-out duration-150"
                    />
                </button>
            </div>
        {/if}
        <div class="text-justify mx-auto">
            <div class="divider" />
            <slot />
        </div>
    </div>
    {#if scrollY > 100}
        <button
            transition:fade
            on:click={() => {
                window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            class="fixed bottom-[5%] right-[5%] md:bottom-[20%] md:right-[10%]"
        >
            <Icon src={AiFillUpCircle} size="32" color="lightblue" />
        </button>
    {/if}
</article>
<div class="h-[10rem]" />

<div class="w-1/2 mx-auto">
    <CommentSection />
</div>

<div class="h-[10rem]" />
