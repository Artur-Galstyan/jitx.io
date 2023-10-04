<script lang="ts">
    import { goto } from "$app/navigation";
    import { page } from "$app/stores";
</script>

<div class="text-center text-lg font-bold my-4">
    ⚠️ This page is still under construction! ⚠️
</div>

<div class="font-extrabold text-lg my-4">Blog Posts</div>
{#each $page.data.posts as post}
    <div class="flex flex-col md:flex-row md:space-x-2">
        <div class="md:w-[30%] flex space-x-4">
            <div class="my-auto text-gray-400">
                {new Date(post.createdAt).toLocaleDateString("en-CA", {
                    year: "numeric",
                    month: "2-digit",
                    day: "2-digit"
                })}
            </div>

            {#if post.status === "DRAFT"}
                <div class="text-warning my-auto">Draft</div>
            {:else if post.status === "PLANNED"}
                <div class="text-info my-auto">Planned</div>
            {/if}
        </div>
        <div class="my-auto flex-1 font-bold flex flex-col">
            <div>
                {post.title}
            </div>
            {#if post.info}
                <div class="text-info text-xs my-2">
                    {post.info}
                </div>
            {/if}
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
