<script lang="ts">
    import Navbar from "$lib/components/Navbar.svelte";
    import "../app.css";

    import { dev } from "$app/environment";
    import { inject } from "@vercel/analytics";
    import CommentSection from "$lib/components/CommentSection.svelte";

    inject({ mode: dev ? "development" : "production" });
</script>

<svelte:head>
    <title>JITx</title>
</svelte:head>
<main class="w-[90%] md:w-1/2 mx-auto">
    <Navbar />
    <slot />
</main>
