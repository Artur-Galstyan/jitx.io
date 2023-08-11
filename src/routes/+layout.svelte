<script lang="ts">
    import Navbar from "$lib/components/Navbar.svelte";
    import "../app.css";
</script>

<svelte:head>
    <title>JITx</title>
</svelte:head>
<main class="w-[80%] mx-auto">
    <Navbar />
    <slot />
</main>
