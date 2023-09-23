<script lang="ts">
    import { invalidateAll } from "$app/navigation";
    import { page } from "$app/stores";
    import CommentSectionEditor from "$lib/components/CommentSectionEditor.svelte";
    import { currentUser } from "$lib/state/currentUser";
    import { errorToast } from "$lib/utils/notifications";
    import { toggleModal } from "$lib/utils/utils";
    import VscSmiley from "svelte-icons-pack/vsc/VscSmiley";
    import Icon from "svelte-icons-pack";

    import { ReactionType } from "@prisma/client";
    let commentToDelete: any = null;

    const reactions = [
        {
            emoji: "👍",
            reactionType: ReactionType.LIKE,
            vote: "upVote"
        },
        {
            emoji: "👎",
            reactionType: ReactionType.DISLIKE,
            vote: "downVote"
        },
        {
            emoji: "🎉",
            reactionType: ReactionType.PARTY,
            vote: "partyVote"
        },
        {
            emoji: "👏",
            reactionType: ReactionType.CLAP,
            vote: "clapVote"
        },
        {
            emoji: "❤️",
            reactionType: ReactionType.HEART,
            vote: "heartVote"
        },
        {
            emoji: "🔥",
            reactionType: ReactionType.FIRE,
            vote: "fireVote"
        },
        {
            emoji: "🙁",
            reactionType: ReactionType.SAD,
            vote: "sadVote"
        }
    ];

    async function submitReaction(
        reaction: { emoji: string; reactionType: ReactionType; vote: string },
        commentId: string
    ) {
        let req = await fetch(
            "/api/comments/" + commentId + "/" + reaction.vote,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                }
            }
        );
        let res = await req.json();

        if (res.error) {
            errorToast("Error", res.error);
        } else {
            console.log(res);
            await invalidateAll();
            const elem = document.activeElement as HTMLElement;
            if (elem) {
                elem?.blur();
            }
        }
    }
</script>

<div class="text-center text-info my-8">Comment Section (WIP)</div>

<CommentSectionEditor />
{#if $page.data.comments.length === 0}
    <div class="text-center text-gray-400">No comments yet</div>
{:else}
    <div class="">
        {#each $page.data.comments as comment}
            <div
                class="flex flex-col border border-solid border-gray-400 p-4 my-2"
            >
                <div class="flex justify-between">
                    <div class="text-gray-400 text-sm flex space-x-2">
                        <div class="avatar">
                            <div class="w-8 rounded">
                                <img
                                    src={comment.User.image}
                                    alt="Tailwind-CSS-Avatar-component"
                                />
                            </div>
                        </div>
                        <div class="my-auto">
                            {comment.User.name} - {new Date(
                                comment.createdAt
                            ).toLocaleString()}
                        </div>
                    </div>
                    <div class="text-gray-400">
                        {#if $currentUser?.id === comment.User.id}
                            <button
                                class="btn btn-xs btn-error"
                                on:click={async () => {
                                    commentToDelete = comment;
                                    toggleModal("are-you-sure-delete");
                                }}
                            >
                                Delete
                            </button>
                        {/if}
                    </div>
                </div>
                <div>
                    {@html comment.body}
                </div>
                <div class="flex space-x-2">
                    <div class="dropdown dropdown-top">
                        <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
                        <!-- svelte-ignore a11y-label-has-associated-control -->
                        <label tabindex="0" class="btn btn-xs btn-outline"
                            ><Icon src={VscSmiley} /></label
                        >
                        <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
                        <div
                            tabindex="0"
                            class="dropdown-content z-[1] p-2 shadow bg-base-100 rounded-box flex space-x-2"
                        >
                            {#each reactions as reaction}
                                <button
                                    on:click={async () => {
                                        await submitReaction(
                                            reaction,
                                            comment.id
                                        );
                                    }}
                                    class="btn btn-xs btn-ghost"
                                    >{reaction.emoji}</button
                                >
                            {/each}
                        </div>
                    </div>
                    <div>
                        {#each reactions as reaction}
                            {#if comment.Reactions.filter((r) => r.type === reaction.reactionType).length > 0}
                                {comment.Reactions.filter(
                                    (r) => r.type === reaction.reactionType
                                ).length}
                                {reaction.emoji}
                            {/if}
                        {/each}
                    </div>
                </div>
            </div>
        {/each}
    </div>
{/if}

<dialog id="are-you-sure-delete" class="modal">
    <div class="modal-box">
        <h3 class="font-bold text-lg">Are you sure?</h3>

        <div class="modal-action">
            <button
                on:click={async (e) => {
                    if (!commentToDelete) return;
                    let req = await fetch("/api/comments", {
                        method: "DELETE",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            commentId: commentToDelete.id
                        })
                    });

                    let res = await req.json();

                    if (res.error) {
                        errorToast("Error", res.error);
                    } else {
                        await invalidateAll();
                        toggleModal("are-you-sure-delete");
                    }
                }}
                class="btn btn-primary">Yes</button
            >
            <button class="btn btn-outline">No</button>
        </div>
    </div>
    <form method="dialog" class="modal-backdrop">
        <button>close</button>
    </form>
</dialog>
