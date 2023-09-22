<script lang="ts">
    import { page } from "$app/stores";

    export let path: string;
    export let caption: string;
    const getStaticFile = (path: string) => {
        return `${$page.url.pathname}/${path}`;
    };
</script>

<figure class="flex flex-col space-y-4">
    <img src={getStaticFile(path)} alt="Example" class="h-full" />
    <div class="text-sm text-gray-400 text-center">{caption}</div>
</figure>
