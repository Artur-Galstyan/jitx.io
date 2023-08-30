<script lang="ts">
    import BsInfoCircle from "svelte-icons-pack/bs/BsInfoCircle";
    import Icon from "svelte-icons-pack";
    export let title: string = "";
    export let content: string;
</script>

<div class="rounded-xl border border-opacity-50 border-info">
    <div class="m-2">
        <div class="text-info flex space-x-8 w-full">
            <Icon src={BsInfoCircle} className="w-6 h-6" />
            <div>
                {title}
            </div>
        </div>
        <p class="text-justify text-info">
            {content}
        </p>
    </div>
</div>
