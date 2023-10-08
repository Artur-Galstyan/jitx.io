<script lang="ts">
    import { goto } from "$app/navigation";
    import { page } from "$app/stores";

    function capitalizeString(str: string) {
        return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
    }
</script>

<div class="text-center text-lg font-bold my-4">
    ⚠️ This page is still under construction! ⚠️
</div>

<div class="overflow-x-auto w-3/4 mx-auto">
    <div class="font-extrabold text-lg my-4">Blog Posts</div>
    <table class="table">
        <!-- head -->
        <thead>
            <tr>
                <th>Created</th>
                <th>Updated</th>
                <th>Title</th>
                <th>Status</th>
                <th>Info</th>
            </tr>
        </thead>
        <tbody>
            {#each $page.data.posts as post}
                <tr
                    class="hover:bg-gray-100 cursor-pointer"
                    on:click={() => goto(`/posts/${post.slug}`)}
                >
                    <td class="text-gray-400"
                        >{new Date(post.createdAt).toLocaleDateString("en-CA", {
                            year: "numeric",
                            month: "2-digit",
                            day: "2-digit"
                        })}</td
                    >
                    <td>
                        {new Date(post.updatedAt).toLocaleDateString("en-CA", {
                            year: "numeric",
                            month: "2-digit",
                            day: "2-digit"
                        })}
                    </td>
                    <td class="font-bold">{post.title}</td>
                    <td
                        class:text-green-400={post.status === "PUBLISHED"}
                        class:text-yellow-400={post.status === "DRAFT"}
                        class:text-blue-400={post.status === "PLANNED"}
                        >{capitalizeString(post.status)}</td
                    >
                    <td>{post.info ?? " "}</td>
                </tr>
            {/each}
        </tbody>
    </table>

    <div class="font-extrabold text-lg my-4">Projects</div>
    <table class="table">
        <!-- head -->
        <thead>
            <tr>
                <th>Last Updated</th>
                <th>Name</th>
                <th>Link</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            {#each $page.data.projects as project}
                <tr
                    class="hover:bg-gray-100 cursor-pointer"
                    on:click={() => goto(`${project.link}`)}
                >
                    <td>
                        {new Date(project.updatedAt).toLocaleDateString(
                            "en-CA",
                            {
                                year: "numeric",
                                month: "2-digit",
                                day: "2-digit"
                            }
                        )}
                    </td>
                    <td class="font-bold">{project.name}</td>

                    <td><a class="link" href={project.link}>Link</a></td>
                    <td>{project.shortDescription}</td>
                </tr>
            {/each}
        </tbody>
    </table>
</div>
