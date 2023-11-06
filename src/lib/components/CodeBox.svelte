<script lang="ts">
    export let filename: string = "";
    export let code: string;
    export let language: string = "python"
</script>

<pre class="code">
    <code class="language-{language}">
        {code}
    </code>
</pre>
<div class="text-center text-gray-400 text-sm">
    {filename}
</div>