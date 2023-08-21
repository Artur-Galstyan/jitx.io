<script lang="ts">
    import { goto } from "$app/navigation";
    import { page } from "$app/stores";
</script>

<div>⚠️ This page is still under construction! ⚠️</div>

<div class="font-extrabold text-lg">Posts</div>
{#each $page.data.posts as post}
    <div class="flex space-x-2">
        <div class="my-auto">
            {post.title}
        </div>
        <button
            class="btn btn-xs btn-info"
            on:click={() => {
                goto(`posts/${post.slug}`);
            }}
        >
            Read
        </button>
    </div>
{/each}
