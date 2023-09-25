<script lang="ts">
    import Navbar from "$lib/components/Navbar.svelte";
    import "../app.css";

    import { dev } from "$app/environment";
    import { inject } from "@vercel/analytics";
    import { currentUser } from "$lib/state/currentUser";
    import { page } from "$app/stores";
    import LoginDialog from "$lib/components/LoginDialog.svelte";

    inject({ mode: dev ? "development" : "production" });

    $currentUser = $page.data.session?.user;
</script>

<svelte:head>
    <title>JITx</title>
    <script src="https://www.google.com/recaptcha/api.js"></script>
</svelte:head>
<main class="w-[90%] md:w-1/2 mx-auto">
    <Navbar />
    <slot />
</main>

<LoginDialog />
