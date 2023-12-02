<script lang="ts">
    import {page} from "$app/stores";

    export let path: string;
    export let caption: string = "";
    const getStaticFile = (path: string) => {
        return `${$page.url.pathname}/${path}`;
    };

    export let width: number = 400;
</script>

<figure class="w-full flex flex-col space-y-4">
    <div class="bg-white p-2 rounded">
        <img
                src={getStaticFile(path)}
                alt={caption}
                class="my-auto mx-auto"
                loading="lazy"
                fetchpriority="low"
                {width}
        />
    </div>

    <div class="text-sm text-gray-400 text-center">{caption}</div>
</figure>
