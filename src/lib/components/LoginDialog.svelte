<script lang="ts">
    import Icon from "svelte-icons-pack"
    import AiFillApple from "svelte-icons-pack/ai/AiFillApple";
    import {page} from "$app/stores";
    import BsDiscord from "svelte-icons-pack/bs/BsDiscord";
    import AiOutlineGithub from "svelte-icons-pack/ai/AiOutlineGithub";
</script>

<dialog id="loginDialogModal" class="modal">
    <form method="dialog" class="modal-box">
        <div class="text-center mb-4 text-xl font-extrabold">Login 🎵</div>
        <div class="w-[80%] mx-auto">
            <div class="flex flex-col justify-center space-y-4">
                <div class="flex flex-col justify-center space-y-4">
                    <button
                            on:click={async () => {
                                await $page.data.supabase.auth.signInWithOAuth({
                                provider: 'github',
                               })
                            }}
                            class="btn btn-ghost"
                    >
                        <span class="w-[4rem] mx-auto">
                            <Icon src={AiOutlineGithub} size="48" color="white"></Icon>
                        </span>

                        <span class="w-[12rem] text-left">
                            Sign in with GitHub</span
                        >
                    </button>
                    <button
                            on:click={async () => {
                                await $page.data.supabase.auth.signInWithOAuth({
                                provider: 'discord',
                               })
                        }}
                            class="btn btn-ghost"
                    >
                        <span class="w-[4rem] mx-auto">
                            <Icon src={BsDiscord} size="48" color="white"></Icon>
                        </span>

                        <span class="w-[12rem] text-left">
                            Sign in with Discord
                        </span>
                    </button>
                    <button
                            on:click={async () => {
                               await $page.data.supabase.auth.signInWithOAuth({
                                provider: 'apple',
                               })
                            }}
                            class="btn btn-ghost"
                    >
                        <span class="w-[4rem] mx-auto">
                            <Icon src={AiFillApple} color="white" size="48"></Icon>
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
