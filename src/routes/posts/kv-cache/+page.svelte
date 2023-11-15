<script lang="ts">

    import AttentionBox from "$lib/components/AttentionBox.svelte";
    import Figure from "$lib/components/Figure.svelte";
    import CodeBox from "$lib/components/CodeBox.svelte";
</script>
<section>
    <h3>
        Introduction
    </h3>
    <p>
        The key-value cache (KV cache) is often used in transformers to cache the results of expensive operations in
        regards to the key and value matrices. Note, that the KV cache is not going to enhance the raw performance of
        your transformer; it will just make it use less computation and, thus, faster. But the outputs itself won't
        improve. In this blog post, we will take a look at how we can implement a KV-cache
        in Jax!
    </p>
</section>

<section>
    <h3>
        How to Implement a KV-cache
    </h3>
    <p>
        Implementing the KV cache in other frameworks is a bit easier than in Jax, because Jax doesn't really like
        dynamic shapes. You will see what I mean by this in just a moment. For now, let's think about the inputs of the
        MultiheadAttention (MHA) layer. The MHA layer gets as input the query, key and value <b>vectors</b> which come
        from the input embedding layer. This is where we need to make the first distinction:
    </p>
    <AttentionBox title="Autoregressive Decoding Only!" content={`The KV cache makes only sense to use when
    you're in an autoregressive decoding setting, where you generate one token at a time!`}/>
    <p>
        Let's clarify this. If you have a decoder-only transformer (like GPT-2), you will generate one token at a time,
        meaning
        that the input to the MHA layer will be a vector of shape <code>(1, embedding_size)</code>. Let's say the first
        token <span class="tooltip" data-tip="We simplify by assuming that one
        token is equal to one word.">is the word "L"</span>. If you pass those through
        your linear layers of your MHA, then the output shapes of the query, key and value will be
        <code>(1, num_heads, embedding_size)</code>. What about the attention weights? Well, they will be of shape
        <code>(num_heads, 1, 1)</code>. This is because we have a single query vector and a single key vector. So far so
        good.
    </p>
    <p>
        Now, let's say we feed the second token "," into the MHA layer. The input will again be of
        shape <code>(1, embedding_size)</code>, but what about our previous token? If we feed one token at a time, we
        won't even know about any of the previous tokens. This is where the KV cache comes into play. At each timestep,
        we get a query as input, which can change all the time (after all, it's the input token), but the key and value
        <i>matrices</i> only change one row at a time!
    </p>
    <p>
        Our first token was "L", so we have a key and value matrix of shape <code>(1, num_heads, embedding_size)</code>.
        Now comes the second token ",". The entire sequence sentence so far is "L,". If we passed in both tokens at
        the same time, then the key and value matrices would be of shape <code>(2, num_heads, embedding_size)</code> but
        more importantly, the first row of the key and value matrix is the same as if we had just passed in the first
        token "L"!. The following figure illustrates this:
    </p>
    <Figure path="KVCache.drawio.svg" caption="The KV cache in action. Notice, how some computations are reused."/>
    <p>
        The figure shows, how the KV cache keeps growing until it reaches <code>max_seq_len</code>. You can also see how
        for the 4th token ("you"), the first 3 rows of the key and value matrix are the same as if we had passed in the
        sequence <i>L, did</i> at once. This is the key idea behind the KV cache. We can reuse the computations of the
        previous tokens, because the key and value matrices only change one row at a time! Then, when it comes to
        computing the attention weights, we can just use the cached key and value matrices and "append" the latest
        key and value vectors to them and then use the result in the attention computation.
    </p>
</section>

<section>
    <h3>Implementing the KV Cache in Jax</h3>
    <p>
        Implementing the KV cache in Jax is a bit more tricky than in other frameworks, because Jax doesn't like dynamic
        shapes. In PyTorch, for instance, it's not a big deal to just append a new row to a matrix. In Jax, however, if
        we
        did that, we would force a re-JIT of the function, which is not what we want. Instead, we need to pre-allocate
        the entire key and value matrix and then just fill it up with the new values.
    </p>
    <p>
        Another thing you should note is that Jax and PyTorch handle states very differently! In PyTorch, which behaves
        in a more pythonic way, you can keep the state within the class itself. Jax, however, follows a functional
        programming paradigm, meaning your functions must not have any side effects (i.e. be pure). In functional
        programming, you
        pass the state as another argument to your function, copy it, modify it and then return it.
    </p>
    <p>
        Thankfully, Equinox has a <code>State</code> API, which we will leverage to implement the KV cache. If you're
        using PyTorch, you can simply keep the state within the class itself.
    </p>
    <p>
        Let's take a look at the code:
        <CodeBox code={`
class MultiheadAttention(eqx.Module):
    ...

    max_seq_len: int = eqx.field(static=True)
    kv_cache_index: Optional[eqx.nn.StateIndex]

    def __init__(self, max_seq_len: int, ...):
        ...
        self.max_seq_len = max_seq_len

        self.kv_cache_index = eqx.nn.StateIndex(
            (
                jnp.zeros(shape=(max_seq_len, num_heads, key_embedding_dim)),
                jnp.zeros(shape=(max_seq_len, num_heads, value_embedding_dim)),
                jnp.zeros(shape=(1,), dtype=jnp.int32),
            )
        )
    def __call__(
        self, x: Float[Array, "seq_len input_dim"], state: Optional[eqx.nn.State]
    ):
        seq_len, _ = x.shape
        key_cache, value_cache, index = None, None, None

        if state is None:
            assert (
                seq_len == self.max_seq_len
            ), "if not autoregressive, seq_len must be max_seq_len"
        else:
            kv_cache = state.get(self.kv_cache_index)
            key_cache, value_cache, index = kv_cache
            index = index[0]
        ...
        if state is not None:
            key_cache = jax.lax.dynamic_update_slice_in_dim(
                key_cache, key_, start_index=index, axis=0
            )
            value_cache = jax.lax.dynamic_update_slice_in_dim(
                value_cache, value, start_index=index, axis=0
            )
        else:
            key_cache = key_
            value_cache = value
        if state is not None:
            new_state = state.set(
                item=self.kv_cache_index,
                value=(key_cache, value_cache, jnp.array([index + 1], dtype=jnp.int32)),
            )

        return output, new_state
        `}/>
    </p>
    <p>
        The relevant part of the code is the update of the key and value cache. We use the
        <code>jax.lax.dynamic_update_slice_in_dim</code> function to update the key and value cache. This function
        updates a matrix at a given index and axis and puts the second argument of the function at the given index.
        The best part about this is that if the index is bigger than the size of the matrix, it will simply wrap around
        and start from the beginning. This is exactly what we want!
    </p>
    <p>
        Once we have the key and value cache, we can simply use it in the attention computation. Note, that the
        query will always be a single token. The idea is that the KV cache keeps a working memory of the previous
        tokens (i.e. the indexing of the previous tokens so to speak) and you only need to compare the current token
        (i.e. the query) against the previous tokens (i.e. the key and value cache).
    </p>
</section>
<section>
    <h3>Final Notes</h3>
    <p>
        You should know that the KV cache is most used in an autoregressive decoder setting, where you generate one
        token at a time and only in inference mode; not during training.
    </p>
    <p>
        The last aspect is in regards to the Equinox State API. To use your stateful model in a batched setting,
        you also need to create a batch of your state. In other words:
        <CodeBox code={`
model, state = eqx.nn.make_with_state(Model)(
    ...
)
states = eqx.filter_vmap(lambda: state, axis_size=batch_size)()

output, states = eqx.filter_vmap(model)(
    first_token, states
)
        `}/>
    </p>
    <p>
        That's it! As you can see, the KV cache is a pretty simple concept, but it can make your transformer much faster
        during inference. Thanks for reading, and I will see you in the next one!
    </p>
</section>
