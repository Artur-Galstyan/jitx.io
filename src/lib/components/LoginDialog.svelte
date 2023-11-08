<script lang="ts">

    import Icon from "svelte-icons-pack"
    import AiFillApple from "svelte-icons-pack/ai/AiFillApple";
    import {goto} from "$app/navigation";
</script>

<dialog id="loginDialogModal" class="modal">
    <form method="dialog" class="modal-box">
        <div class="text-center mb-4 text-xl font-extrabold">Login 🎵</div>
        <div class="w-[80%] mx-auto">
            <div class="flex flex-col justify-center space-y-4">
                <div class="flex flex-col justify-center space-y-4">
                    <button
                            on:click={async () => {
                                await goto("/login/github");
                            }}
                            class="btn btn-ghost"
                    >
                        <span class="w-[4rem]">
                            <img
                                    class="mx-auto w-[46px]"
                                    src="/buttons/github.svg"
                                    alt="github Icon"
                            />
                        </span>

                        <span class="w-[12rem] text-left">
                            Sign in with GitHub</span
                        >
                    </button>
                    <button
                            on:click={async () => {
                               await goto("/login/discord")
                        }}
                            class="btn btn-ghost"
                    >
                        <span class="w-[4rem]">
                            <img
                                    class="mx-auto"
                                    src="/buttons/discord.svg"
                                    alt="Discord Icon"
                            />
                        </span>

                        <span class="w-[12rem] text-left">
                            Sign in with Discord
                        </span>
                    </button>
                    <button
                            on:click={async () => {
                               await goto("/login/discord")
                        }}
                            class="btn btn-ghost"
                    >
                        <span class="">
                            <Icon src={AiFillApple} size="48"></Icon>
                        </span>

                        <span class="w-[12rem] text-left">
                            Sign in with Apple
                        </span>
                    </button>
                </div>
            </div>
        </div>
    </form>
    <form method="dialog" class="modal-backdrop">
        <button/>
    </form>
</dialog>
