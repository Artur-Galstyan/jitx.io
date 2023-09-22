<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { Editor } from "@tiptap/core";
    import StarterKit from "@tiptap/starter-kit";
    import AiOutlineUnorderedList from "svelte-icons-pack/ai/AiOutlineUnorderedList";
    import Placeholder from "@tiptap/extension-placeholder";
    import AiOutlineOrderedList from "svelte-icons-pack/ai/AiOutlineOrderedList";
    import AiOutlineCode from "svelte-icons-pack/ai/AiOutlineCode";
    import AiOutlineUndo from "svelte-icons-pack/ai/AiOutlineUndo";
    import AiOutlineRedo from "svelte-icons-pack/ai/AiOutlineRedo";
    import Icon from "svelte-icons-pack";
    let element: HTMLDivElement;
    let editor: Editor;

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
    });

    onDestroy(() => {
        if (editor) {
            editor.destroy();
        }
    });
</script>

<div class="text-center text-info">Comment Section (WIP)</div>
<div class="w-1/2 mx-auto border border-solid border-gray-600 p-8">
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

                <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
                <div class="dropdown">
                    <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
                    <!-- svelte-ignore a11y-label-has-associated-control -->
                    <label tabindex="0" class="m-1 btn btn-sm btn-outline"
                        >Headings</label
                    >
                    <ul
                        tabindex="0"
                        class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-52"
                    >
                        <li>
                            ><button
                                on:click={() =>
                                    editor
                                        .chain()
                                        .focus()
                                        .toggleHeading({ level: 1 })
                                        .run()}
                                class={editor.isActive("heading", {
                                    level: 1
                                })
                                    ? "is-active"
                                    : ""}
                            >
                                <h1>H1</h1>
                            </button>
                        </li>
                        <li>
                            <button
                                on:click={() =>
                                    editor
                                        .chain()
                                        .focus()
                                        .toggleHeading({ level: 2 })
                                        .run()}
                                class={editor.isActive("heading", {
                                    level: 2
                                })
                                    ? "is-active"
                                    : ""}
                            >
                                <h2>H2</h2>
                            </button>
                        </li>
                        <li>
                            <button
                                on:click={() =>
                                    editor
                                        .chain()
                                        .focus()
                                        .toggleHeading({ level: 3 })
                                        .run()}
                                class={editor.isActive("heading", {
                                    level: 3
                                })
                                    ? "is-active"
                                    : ""}
                            >
                                <h3>H3</h3>
                            </button>
                        </li>
                        <li>
                            <button
                                on:click={() =>
                                    editor
                                        .chain()
                                        .focus()
                                        .toggleHeading({ level: 4 })
                                        .run()}
                                class={editor.isActive("heading", {
                                    level: 4
                                })
                                    ? "is-active"
                                    : ""}
                            >
                                <h4>H4</h4>
                            </button>
                        </li>
                    </ul>
                </div>

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
                    on:click={() =>
                        editor.chain().focus().setHorizontalRule().run()}
                >
                    hr
                </button>
                <button
                    on:click={() => editor.chain().focus().setHardBreak().run()}
                >
                    Line break
                </button>
                <button
                    on:click={() => editor.chain().focus().undo().run()}
                    disabled={!editor.can().chain().focus().undo().run()}
                >
                    <Icon src={AiOutlineUndo} />
                </button>
                <button
                    on:click={() => editor.chain().focus().redo().run()}
                    disabled={!editor.can().chain().focus().redo().run()}
                >
                    <Icon src={AiOutlineRedo} />
                </button>
            </div>
        </div>
    {/if}
    <div class="divider" />
    <div class="">
        <div class="prose" bind:this={element} />
    </div>
</div>

<style>
    button.active {
        background: black;
        color: white;
    }
</style>
