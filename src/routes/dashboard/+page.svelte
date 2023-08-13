<script>
    import { enhance } from "$app/forms";
    import { page } from "$app/stores";
    import { errorToast, successNotification } from "$lib/utils/notifications";
    import { toggleModal } from "$lib/utils/utils";
</script>

<div>
    <div class="font-extrabold text-lg">Posts</div>
    {#each $page.data.posts as post}
        <div class="flex justify-between">
            <div>Title {post.title}</div>
            <div>
                Created at {new Date(post.createdAt)
                    .toISOString()
                    .split("T")[0]}
            </div>
            <div>Slug {post.slug}</div>
            <div>Tags {post.tags}</div>
            <form
                action="?/delete"
                method="post"
                class="flex space-x-4"
                use:enhance
            >
                <div>Id {post.id}</div>
                <input type="hidden" name="id" value={post.id} />
                <div>
                    <button class="btn btn-xs btn-warning"> Delete </button>
                </div>
            </form>
        </div>
    {/each}
</div>

<button
    on:click={() => {
        toggleModal("create-post-dialog");
    }}
    class="btn">Create Post</button
>
<dialog id="create-post-dialog" class="modal">
    <form method="dialog" class="modal-box w-11/12 max-w-5xl">
        <h3 class="font-bold text-lg">Create Post</h3>
        <form
            action="?/create"
            method="post"
            use:enhance={({
                formElement,
                formData,
                action,
                cancel,
                submitter
            }) => {
                // `formElement` is this `<form>` element
                // `formData` is its `FormData` object that's about to be submitted
                // `action` is the URL to which the form is posted
                // calling `cancel()` will prevent the submission
                // `submitter` is the `HTMLElement` that caused the form to be submitted

                return async ({ result, update }) => {
                    console.log(result);
                    if (result.type === "error")
                        errorToast("Error", result.error);
                    else if (result.type === "success") {
                        toggleModal("create-post-dialog");
                        successNotification("Success", "Post created");
                    }
                    // `result` is an `ActionResult` object
                    // `update` is a function which triggers the default logic that would be triggered if this callback wasn't set
                    update();
                };
            }}
        >
            <div class="flex flex-col space-y-4">
                <div>
                    <label for="title">Title</label>
                    <input
                        type="text"
                        name="title"
                        id="title"
                        class="input input-primary w-full"
                    />
                </div>
                <div>
                    <label for="title">Slug</label>
                    <input
                        type="text"
                        name="slug"
                        id="slug"
                        class="input input-primary w-full"
                    />
                </div>
                <div>
                    <label for="title">Tags</label>
                    <input
                        type="text"
                        name="tags"
                        id="tags"
                        class="input input-primary w-full"
                    />
                </div>
            </div>
            <div class="flex justify-center">
                <button class="btn btn-primary" type="submit">Create</button>
            </div>
        </form>
    </form>
    <form method="dialog" class="modal-backdrop">
        <button>close</button>
    </form>
</dialog>
