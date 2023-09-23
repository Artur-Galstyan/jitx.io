<script lang="ts">
    import { goto } from "$app/navigation";
    import { page } from "$app/stores";
</script>

<div class="text-center text-lg font-bold">
    ⚠️ This page is still under construction! ⚠️
</div>

<div class="font-extrabold text-lg">Posts</div>
{#each $page.data.posts as post}
    <div class="flex space-x-2">
        {#if post.draft}
            <div class="text-warning">Draft</div>
        {/if}
        <div class="my-auto">
            {new Date(post.createdAt).toLocaleDateString("en-CA", {
                year: "numeric",
                month: "2-digit",
                day: "2-digit"
            })}
        </div>
        <div class="my-auto flex-1">
            {post.title}
        </div>
        <button
            class="btn btn-xs btn-info btn-outline my-auto"
            on:click={() => {
                goto(`posts/${post.slug}`);
            }}
        >
            Read
        </button>
    </div>
    <div class="flex space-x-2 my-2">
        {#each post.tags as tag}
            <div class="badge badge-outline badge-sm">
                {tag}
            </div>
        {/each}
    </div>
    <div class="divider" />
{/each}
