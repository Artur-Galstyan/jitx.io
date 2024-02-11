<script lang="ts">
  import Katex from "$lib/components/Katex.svelte";
</script>

<section>
  <h2>Introduction</h2>
  <p>
    Proximal Policy Optimization (PPO) is one of the most successful RL
    algorithms out there. It is known for its ease of implementation and
    excellent performance. Today, we will cover PPO and you will see how to
    implement it yourself using Jax! So, let's get started!
  </p>
</section>
<section>
  First, where does PPO fall into the RL algorithm buckets? PPO is an
  Actor-Critic algorithm, which means it falls into the policy gradient faction.
  In fact, it's a natural extension to the vanilla policy gradient algorithm as
  we will see in a moment. First, let's review the vanilla policy gradient
  algorithm (which we had already covered in <a
    href="/posts/reinforce-policy-gradient"
    data-sveltekit-reload>this blogpost</a
  > before).
</section>
<section>
  <p>
    First, we define the reward (infinite horizon and discounted) given some
    trajectory <Katex
      math={`R(\\tau) = \\sum_{t=0}^\\infty \\gamma^t r_t`}
      displayMode
    />
  </p>
  <p>
    Next, what is the probability to sample that particular trajectory? We
    define it as
    <Katex
      math={`P(\\tau | \\pi_\\theta) = p_0(s_0) \\prod_{t = 0} ^ \\infty p(s_{t+1} | s_t, a_t) \\pi_\\theta(a_t | s_t)`}
      displayMode
    />
  </p>
  <p>
    Spoken out loud, the above equation translates to <i
      >the probability to sample a particular trajectory <Katex
        math={"\\tau"}
      /> given a policy <Katex math={"\\pi"} /> with parameters <Katex
        math={"\\theta"}
      /> is equal to the the probability of the initial state times the product of
      the probability to land in a state <Katex math={"s_{t+1}"} /> given the current
      state <Katex math={"s_t"} /> and the action <Katex math={"a_t"} /> times the
      probability to pick the action <Katex math={"a_t"} /> given the current state
      <Katex math={"s_t"} />
    </i>.
  </p>
  <p>
    Now that we now the probability to sample a trajectory given our current
    policy, we define the objective function as <Katex
      math={`J(\\pi_\\theta) = \\mathbb{E}_{\\tau\\sim\\pi_\\theta} R(\\tau) = \\int_t P(\\tau | \\pi_\\theta) R(\\tau)`}
      displayMode
    />
  </p>
  <p>
    Given that objective function, we next computed the gradient of the
    objective function. I won't go into the details here, but if you want to
    understand the details, I suggest you read my other <a
      href="/posts/reinforce-policy-gradient">blog post</a
    >. Anyway, the gradient of the objective function is defined as <Katex
      math={`\\nabla_\\theta J(\\pi_\\theta) = \\mathbb{E}_{\\tau\\sim\\pi_\\theta} \\left [ R(\\tau) \\sum_t \\nabla_\\theta \\pi_\\theta (a_t | s_t) \\right ]`}
      displayMode
    />
  </p>
</section>
