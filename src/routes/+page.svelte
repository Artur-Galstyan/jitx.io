<script lang="ts">
    import { goto } from "$app/navigation";
    import { page } from "$app/stores";
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
</script>

<div class="overflow-x-auto w-3/4 mx-auto">
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
    <table class="table">
        <!-- head -->
        <thead>
            <tr>
                <th class="hidden md:table-cell">Created</th>
                <th class="hidden md:table-cell">Updated</th>
                <th class="text-center md:text-left">Title</th>
                <th class="hidden md:table-cell">Status</th>
            </tr>
        </thead>
        <tbody>
            {#each posts as post}
                <tr
                    class="hover:bg-gray-100 cursor-pointer"
                    on:click={() => goto(`/posts/${post.slug}`)}
                >
                    <td class="text-gray-400 hidden md:table-cell"
                        >{new Date(post.createdAt).toLocaleDateString("en-CA", {
                            year: "numeric",
                            month: "2-digit",
                            day: "2-digit"
                        })}</td
                    >
                    <td class="hidden md:table-cell">
                        {new Date(post.updatedAt).toLocaleDateString("en-CA", {
                            year: "numeric",
                            month: "2-digit",
                            day: "2-digit"
                        })}
                    </td>
                    <td class="font-bold text-center md:text-justify"
                        >{post.title}</td
                    >
                    <td
                        class="hidden md:table-cell"
                        class:text-green-400={post.status === "PUBLISHED"}
                        class:text-yellow-400={post.status === "DRAFT"}
                        class:text-blue-400={post.status === "PLANNED"}
                        >{capitalizeString(post.status)}</td
                    >
                </tr>
            {/each}
        </tbody>
    </table>

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
