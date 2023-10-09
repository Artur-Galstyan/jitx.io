<script lang="ts">
    import Navbar from "$lib/components/Navbar.svelte";
    import "../app.css";
    import { dev } from "$app/environment";
    import { inject } from "@vercel/analytics";
    import { currentUser } from "$lib/state/currentUser";
    import { page } from "$app/stores";
    import LoginDialog from "$lib/components/LoginDialog.svelte";
    import "@fontsource/fira-code";
    import { onNavigate } from "$app/navigation";
    import { transisting } from "$lib/state/transisting";

    onNavigate((navigation) => {
        if (!document.startViewTransition) return;
        $transisting = true;
        return new Promise((resolve) => {
            document.startViewTransition(async () => {
                resolve();
                await navigation.complete;
                $transisting = false;
            });
        });
    });

    inject({ mode: dev ? "development" : "production" });

    $currentUser = $page.data.session?.user;
</script>

<svelte:head>
    <title>JITx</title>
    <script src="https://www.google.com/recaptcha/api.js"></script>
</svelte:head>
<main class="mx-auto">
    <Navbar />
    <slot />
</main>

<LoginDialog />
