<script lang="ts">
  import HintBox from "$lib/components/HintBox.svelte";
  import Figure from "$lib/components/Figure.svelte";
  import CodeBox from "$lib/components/CodeBox.svelte";
</script>

<section>
  <h3>Introduction</h3>
  <p>
    Imagine, you developed a new AI LLM and wanted to deploy that to the web.
    You already read my previous <a
      href="https://www.jitx.io/posts/ml-deployment">blogpost</a
    > and already have quite a bit setup but you're running into a problem. For one,
    you don't have any monitoring, which is already bad, but furthermore, you decided
    to seperate your worker code from the API code, perhaps by writing the API in
    a statically typed language like Rust.
  </p>
  <p>
    The issue is now: how does your API (which has no access to the Python
    worker source code) talk with your workers? And how can you monitor
    everything?
  </p>
  <p>
    In this blog post, you will learn exactly that! You will learn how to deploy
    a "Celery API" and how you can monitor your workers. And you will also learn
    how to dockerize everything such that you can use Kubernetes at some point
    in the future to automatically scale everything!
  </p>
  <p class="text-xs">
    Although we won't cover how to use Kubernetes in this blog post; maybe
    another time :)
  </p>
</section>
