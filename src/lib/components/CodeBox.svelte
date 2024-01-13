<script lang="ts">
  import { successNotification } from "$lib/utils/notifications";
  import Icon from "svelte-icons-pack";
  import AiOutlineCopy from "svelte-icons-pack/ai/AiOutlineCopy";
  export let filename: string = "";
  export let code: string;
  export let language: string = "python";
</script>

<div class="relative">
  <div class="absolute top-[32px] right-4">
    <button
      on:click={() => {
        navigator.clipboard.writeText(code);
        successNotification("Success", "Copied to clipboard");
      }}
      class="btn btn-ghost btn-sm"
    >
      <Icon src={AiOutlineCopy} size="32" color="white" />
    </button>
  </div>
  <pre class="code">
    <code class="language-{language}">
        {code}
    </code>
</pre>
  <div class="text-center text-gray-400 text-sm">
    {filename}
  </div>
</div>
