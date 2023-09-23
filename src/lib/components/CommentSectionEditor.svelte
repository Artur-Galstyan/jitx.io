<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { Editor } from "@tiptap/core";
    import StarterKit from "@tiptap/starter-kit";
    import AiOutlineUnorderedList from "svelte-icons-pack/ai/AiOutlineUnorderedList";
    import Placeholder from "@tiptap/extension-placeholder";
    import AiOutlineOrderedList from "svelte-icons-pack/ai/AiOutlineOrderedList";
    import AiOutlineCode from "svelte-icons-pack/ai/AiOutlineCode";
    import Icon from "svelte-icons-pack";
    import GoogleReCaptchaDisclaimer from "$lib/components/GoogleReCaptchaDisclaimer.svelte";
    import { page } from "$app/stores";
    import { currentUser } from "$lib/state/currentUser";
    import { showLoginDialog, toggleModal } from "$lib/utils/utils";
    import { signOut } from "@auth/sveltekit/client";
    import { errorToast, successNotification } from "$lib/utils/notifications";
    import { PUBLIC_CAPTCHA_SITE_KEY } from "$env/static/public";
    import { invalidateAll } from "$app/navigation";
    let element: HTMLDivElement;
    let editor: Editor;

    let grecaptcha: any;
    let loading = false;
    onMount(() => {
        editor = new Editor({
            element: element,
            extensions: [
                StarterKit,
                Placeholder.configure({
                    placeholder: "Leave a comment ✨"
                })
            ],

            onTransaction: () => {
                editor = editor;
            }
        });

        grecaptcha = (window as any).grecaptcha || {};

        grecaptcha.ready(function () {
            grecaptcha.render("grecaptcha", {
                sitekey: PUBLIC_CAPTCHA_SITE_KEY,
                size: "invisible"
            });
        });
    });

    onDestroy(() => {
        if (editor) {
            editor.destroy();
        }
    });

    async function submitComment(token: string) {
        const content = editor.getHTML();

        let req = await fetch("/api/comments", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                body: content,
                postId: $page.data.post.id,
                userId: $currentUser?.id,
                token: token
            })
        });
        let res = await req.json();

        if (res.error) {
            errorToast("Error", res.error);
        } else {
            editor.chain().focus().clearContent().run();
            loading = false;
            toggleModal("are-you-sure");
            successNotification("Success", "Comment posted!");

            await invalidateAll();
        }
    }
</script>

<div class="w-full mx-auto border border-solid border-gray-400 p-4">
    {#if editor}
        <div class="">
            <div class="flex space-x-2 my-auto">
                <button
                    on:click={() => editor.chain().focus().toggleBold().run()}
                    disabled={!editor.can().chain().focus().toggleBold().run()}
                    class={editor.isActive("bold") ? "is-active" : ""}
                >
                    <b>B</b>
                </button>
                <button
                    on:click={() => editor.chain().focus().toggleItalic().run()}
                    disabled={!editor
                        .can()
                        .chain()
                        .focus()
                        .toggleItalic()
                        .run()}
                    class={editor.isActive("italic") ? "is-active" : ""}
                >
                    <i>I</i>
                </button>
                <button
                    on:click={() => editor.chain().focus().toggleStrike().run()}
                    disabled={!editor
                        .can()
                        .chain()
                        .focus()
                        .toggleStrike()
                        .run()}
                    class={editor.isActive("strike") ? "is-active" : ""}
                >
                    <s>S</s>
                </button>
                <button
                    on:click={() => editor.chain().focus().toggleCode().run()}
                    disabled={!editor.can().chain().focus().toggleCode().run()}
                    class={editor.isActive("code") ? "is-active" : ""}
                >
                    <code>{"</>"}</code>
                </button>

                <button
                    on:click={() => editor.chain().focus().setParagraph().run()}
                    class={editor.isActive("paragraph") ? "is-active" : ""}
                >
                    P
                </button>

                <button
                    on:click={() =>
                        editor.chain().focus().toggleBulletList().run()}
                    class={editor.isActive("bulletList") ? "is-active" : ""}
                >
                    <Icon src={AiOutlineUnorderedList} />
                </button>
                <button
                    on:click={() =>
                        editor.chain().focus().toggleOrderedList().run()}
                    class={editor.isActive("orderedList") ? "is-active" : ""}
                >
                    <Icon src={AiOutlineOrderedList} />
                </button>
                <button
                    on:click={() =>
                        editor.chain().focus().toggleCodeBlock().run()}
                    class={editor.isActive("codeBlock") ? "is-active" : ""}
                >
                    <Icon src={AiOutlineCode} />
                </button>
                <button
                    on:click={() =>
                        editor.chain().focus().toggleBlockquote().run()}
                    class={editor.isActive("blockquote") ? "is-active" : ""}
                >
                    <blockquote>“”</blockquote>
                </button>
                <button
                    class=""
                    on:click={() =>
                        editor.chain().focus().setHorizontalRule().run()}
                >
                    hr
                </button>
                <div class="flex-1" />

                {#if $currentUser}
                    <div class="flex flex-col md:flex-row md:space-x-2">
                        <button
                            on:click={() => {
                                toggleModal("are-you-sure");
                            }}
                            class="btn btn-primary btn-sm my-1 md:my-0"
                        >
                            Submit
                        </button>

                        <button
                            on:click={() => {
                                currentUser.set(undefined);
                                signOut();
                            }}
                            class="btn btn-outline btn-xs my-auto"
                            >Sign Out</button
                        >
                    </div>
                {:else}
                    <button
                        on:click={() => {
                            showLoginDialog();
                        }}
                        class="btn btn-primary btn-outline btn-sm"
                    >
                        Login to comment
                    </button>
                {/if}
            </div>
        </div>
        <div class="divider" />
    {:else}
        <div class="flex justify-center">
            <span class="loading loading-ball loading-sm" />
        </div>
    {/if}
    <div class="">
        <div class="prose" bind:this={element} />
    </div>
</div>
<div id="grecaptcha" />
<dialog id="are-you-sure" class="modal">
    <div class="modal-box">
        <h3 class="font-bold text-lg">Are you sure?</h3>

        <div class="modal-action">
            <button
                disabled={loading}
                on:click={async (e) => {
                    loading = true;
                    e.preventDefault();

                    grecaptcha
                        .execute({ action: "submit" })
                        .then(function (token) {
                            submitComment(token);
                        });
                }}
                class="btn btn-primary">Yes</button
            >
            <button class="btn btn-outline">No</button>
        </div>
        <GoogleReCaptchaDisclaimer />
    </div>
    <form method="dialog" class="modal-backdrop">
        <button>close</button>
    </form>
</dialog>
