<script lang="ts">
    import {
        goto,
        invalidate,
        invalidateAll,
        preloadCode,
        preloadData
    } from "$app/navigation";
    import { page } from "$app/stores";
    import Hero from "$lib/components/Hero.svelte";
    import { POSTS_PER_PAGE } from "$lib/utils/constants";
    import Fuse from "fuse.js";
    function capitalizeString(str: string) {
        return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
    }

    const fuseOptions = {
        // isCaseSensitive: false,
        // includeScore: false,
        // shouldSort: true,
        // includeMatches: false,
        // findAllMatches: false,
        // minMatchCharLength: 1,
        // location: 0,
        threshold: 0.2,
        // distance: 100,
        // useExtendedSearch: false,
        // ignoreLocation: false,
        // ignoreFieldNorm: false,
        // fieldNormWeight: 1,
        keys: ["title", "tags"]
    };

    const fuse = new Fuse($page.data.posts, fuseOptions);
    let posts = $page.data.posts;

    let pageSize = POSTS_PER_PAGE;
    $: totalPosts = $page.data.totalPosts;
    $: totalPages = Math.ceil(totalPosts / pageSize);
    $: currentPage = 1;
</script>

<div class="overflow-x-auto xl:w-3/4 mx-auto">
    <Hero />
    <div class="flex justify-between">
        <div class="font-extrabold text-lg my-4">Blog Posts</div>
        <input
            type="text"
            class="my-auto input input-sm"
            placeholder="Search 🔎"
            on:input={(e) => {
                const query = e.target.value;
                if (query.length > 0) {
                    posts = fuse.search(query).map((result) => result.item);
                } else {
                    posts = $page.data.posts;
                }
            }}
        />
    </div>

    <div class="grid grid-cols-1 gap-x-2 gap-y-2">
        {#each $page.data.posts as post, i}
            <div
                tabindex={i}
                role="button"
                on:mouseenter={() => {
                    preloadData(`/posts/${post.slug}`);
                }}
                on:keydown={() => {}}
                on:keyup={() => {}}
                on:keypress={() => {}}
                on:click={async () => await goto(`/posts/${post.slug}`)}
                class="card card-side rounded-xl hover:bg-accent hover:bg-opacity-40 transition duration-150 ease-in-out"
            >
                {#if post.thumbnail}
                    <figure class="mx-auto w-[30%] md:w-[10rem] p-1 rounded-xl">
                        <img
                            src={"/posts/" + post.slug + "/thumbnail.webp"}
                            alt={post.title}
                            class="rounded-xl mx-auto w-full"
                        />
                    </figure>
                {/if}
                <div class="card-body px-4 py-4 my-0 w-full">
                    <h2 class="card-title my-0 py-0 font-extrabold">
                        {post.title}
                    </h2>
                    <p>{post.shortDescription}</p>
                    <div class="card-actions">
                        <div
                            class="my-auto flex-1 text-gray-400 text-xs flex flex-col"
                        >
                            <div>
                                <span class=" ">Last Updated</span>
                                {new Date(post.updatedAt).toLocaleDateString(
                                    "en-CA",
                                    {
                                        year: "numeric",
                                        month: "2-digit",
                                        day: "2-digit"
                                    }
                                )}
                            </div>

                            <div
                                class:text-green-400={post.status ===
                                    "PUBLISHED"}
                                class:text-yellow-400={post.status === "DRAFT"}
                                class:text-blue-400={post.status === "PLANNED"}
                            >
                                {capitalizeString(post.status)}
                            </div>
                        </div>
                        <a href={"/posts/" + post.slug}>
                            <button class="btn btn-primary btn-sm">Read</button>
                        </a>
                    </div>
                </div>
            </div>
            <div class="divider w-1/2 mx-auto" />
        {/each}
    </div>
    <div class:hidden={totalPages <= 1} class="join grid grid-cols-2">
        <button
            class:btn-disabled={currentPage <= 1}
            on:click={async () => {
                currentPage--;
                await goto("/?skip=" + (currentPage - 1) * pageSize);
                await invalidateAll();
            }}
            class="join-item btn btn-outline">Previous page</button
        >
        <button
            class:btn-disabled={currentPage === totalPages}
            on:click={async () => {
                currentPage++;
                await goto("/?skip=" + (currentPage - 1) * pageSize);
                await invalidateAll();
            }}
            class="join-item btn btn-outline">Next</button
        >
    </div>
    <div class="font-extrabold text-lg my-4">Projects</div>
    <table class="table">
        <!-- head -->
        <thead>
            <tr>
                <th class="hidden md:table-cell">Last Updated</th>
                <th class="text-center md:text-left">Name</th>
                <th class="hidden md:table-cell">Link</th>
                <th class="hidden md:table-cell">Description</th>
            </tr>
        </thead>
        <tbody>
            {#each $page.data.projects as project}
                <tr
                    class="hover:bg-gray-100 cursor-pointer"
                    on:click={() => goto(`${project.link}`)}
                >
                    <td class="hidden md:table-cell">
                        {new Date(project.updatedAt).toLocaleDateString(
                            "en-CA",
                            {
                                year: "numeric",
                                month: "2-digit",
                                day: "2-digit"
                            }
                        )}
                    </td>
                    <td class="font-bold text-center md:text-justify"
                        >{project.name}</td
                    >

                    <td class="hidden md:table-cell"
                        ><a class="link" href={project.link}>Link</a></td
                    >
                    <td class="hidden md:table-cell"
                        >{project.shortDescription}</td
                    >
                </tr>
            {/each}
        </tbody>
    </table>
</div>
