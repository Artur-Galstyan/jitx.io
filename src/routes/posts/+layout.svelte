<script lang="ts">
    import python from "highlight.js/lib/languages/python";
    import hljs from "highlight.js";
    import { onMount } from "svelte";
    import type { Post } from "@prisma/client";
    import { page } from "$app/stores";
    import Icon from "svelte-icons-pack";
    import AiOutlineGithub from "svelte-icons-pack/ai/AiOutlineGithub";

    hljs.registerLanguage("python", python);
    const post: Post = $page.data.post;

    onMount(() => {
        hljs.highlightAll();
    });
</script>

<svelte:head />
<main class="flex justify-center">
    <div class="w-[95%] md:w-[90%] mx-auto">
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
        <slot />
    </div>
</main>
<div class="h-[10rem]" />
