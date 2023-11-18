<script lang="ts">
    import BsInfoCircle from "svelte-icons-pack/bs/BsInfoCircle";
    import Icon from "svelte-icons-pack";

    export let title: string = "Hint";
    export let content: string;
</script>

<div class="rounded-b-xl border border-opacity-50 border-info my-2">
    <div class="">
        <div class=" flex space-x-8 w-full bg-blue-400 p-2">
            <Icon src={BsInfoCircle} className="w-6 h-6"/>
            <div>
                {title}
            </div>
        </div>
        <p class="text-justify text-info p-1">
            {@html content}
        </p>
    </div>
</div>
