<script lang="ts">

    import BsExclamationTriangle from "svelte-icons-pack/bs/BsExclamationTriangle";
    import Icon from "svelte-icons-pack";

    export let title: string = "";
    export let content: string;
</script>

<div class="rounded-xl border border-opacity-50 border-warning my-2">
    <div class="m-2">
        <div class="text-warning flex space-x-8 w-full">
            <Icon src={BsExclamationTriangle} className="w-6 h-6" />
            <div>
                {title}
            </div>
        </div>
        <p class="text-justify text-warning">
            {@html content}
        </p>
    </div>
</div>
