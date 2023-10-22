<script>
    import { transisting } from "$lib/state/transisting";
    import { fade } from "svelte/transition";
</script>

<nav class="flex">
    <div class="flex flex-1">
        <a
            data-sveltekit-preload-data
            class="cursor-pointer hover:bg-blue-600 hover:bg-opacity-20 transition duration-150 ease-in-out rounded-xl m-4"
            href="/"
        >
            <img src="/logo1.webp" alt="Logo" class="w-36 p-1" />
        </a>
        <div class="my-auto">
            <span
                class="header-text font-extrabold text-transparent text-5xl bg-clip-text bg-gradient-to-r from-amber-400 to-pink-600"
                >JIT</span
            >
            <span class="header-text font-extrabold text-4xl text-sky-500">
                x
            </span>

            <div class="my-auto italic">
                AI Projects and Software Engineering
            </div>
            <div class="text-xs text-gray-400">by Artur A. Galstyan</div>
        </div>
    </div>
    {#if $transisting}
        <div>
            <span transition:fade class="loading loading-spinner loading-lg" />
        </div>
    {/if}
</nav>
