<script>
  import { transisting } from "$lib/state/transisting";
  import { fade } from "svelte/transition";
  import Icon from "svelte-icons-pack";
  import FaBrandsTwitter from "svelte-icons-pack/fa/FaBrandsTwitter";
  import FaBrandsLinkedin from "svelte-icons-pack/fa/FaBrandsLinkedin";
  import FaBrandsGithub from "svelte-icons-pack/fa/FaBrandsGithub";
</script>

<nav class="flex">
  <div class="flex flex-1">
    <a
      data-sveltekit-preload-data
      class="cursor-pointer hover:bg-blue-600 hover:bg-opacity-20 transition duration-150 ease-in-out rounded-xl m-4"
      href="/"
    >
      <img src="/littlehelper.svg" alt="Logo" class="w-36 p-1" />
    </a>
    <div class="my-auto">
      <span
        class="header-text font-extrabold text-transparent text-5xl bg-clip-text bg-gradient-to-r from-amber-400 to-pink-600"
        >JIT</span
      >
      <span class="header-text font-extrabold text-4xl text-sky-500"> x </span>

      <div class="my-auto italic">AI Projects and Software Engineering</div>
      <div class="text-xs text-gray-400">by Artur A. Galstyan</div>
    </div>
  </div>
  <div class="my-auto flex justify-between space-x-4 mx-8">
    <button
      on:click={() => {
        window.location.href = "https://twitter.com/ArturAGalstyan";
      }}
      class="btn btn-ghost bg-white hover:bg-white"
    >
      <Icon src={FaBrandsTwitter} color="#1DA1F2" className="w-8 h-8 my-auto" />
    </button>
    <button
      on:click={() => {
        window.location.href =
          "https://www.linkedin.com/in/artur-galstyan-262842204/";
      }}
      class="btn btn-ghost bg-white hover:bg-white"
    >
      <Icon
        src={FaBrandsLinkedin}
        color="#0077b5"
        className="w-8 h-8 my-auto"
      />
    </button>
    <button
      on:click={() => {
        window.location.href = "https://www.github.com/artur-galstyan";
      }}
      class="btn btn-ghost bg-white hover:bg-white"
    >
      <Icon src={FaBrandsGithub} color="black" className="w-8 h-8 my-auto" />
    </button>
  </div>
  {#if $transisting}
    <div>
      <span transition:fade class="loading loading-spinner loading-lg" />
    </div>
  {/if}
</nav>
