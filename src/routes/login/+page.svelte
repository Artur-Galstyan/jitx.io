<script lang="ts">
    import { page } from "$app/stores";
    import { signIn, signOut } from "@auth/sveltekit/client";
    console.log($page.data.session);
</script>

{#if !$page.data.session?.user}
    <button
        class="btn btn-primary"
        on:click={() => {
            signIn("discord");
        }}
    >
        Sign in (but only me!)
    </button>
{:else}
    <div>
        Hi {$page.data.session.user.name}
    </div>
    <button class="btn btn-primary" on:click={signOut}> Sign out </button>
{/if}
