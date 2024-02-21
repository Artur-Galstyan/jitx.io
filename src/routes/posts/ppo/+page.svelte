<script lang="ts">
  import Figure from "$lib/components/Figure.svelte";
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
      math={`\\nabla_\\theta J(\\pi_\\theta) = \\mathbb{E}_{\\tau\\sim\\pi_\\theta} \\left [ R(\\tau) \\sum_t \\nabla_\\theta \\log \\pi_\\theta (a_t | s_t) \\right ]`}
      displayMode
    />
  </p>
  <p>
    Furthermore, we had also improved this objective function by including a
    bias term in the form of a value network, like so:

    <Katex
      math={`\\nabla_{\\theta} J(\\pi_{\\theta}) = \\underset{\\tau \\sim \\pi_{\\theta}}{E}{\\sum_{t=0}^{T} \\nabla_{\\theta} \\log \\pi_{\\theta}(a_t |s_t) \\left(\\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\\right)}`}
      displayMode={true}
    />
  </p>
  <p>
    With this gradient, we can update the parameters of our policy, which in
    turn should increase our rewards. In the update function, we had also
    included a learning rate <Katex math={"\\alpha"} /> to make our step sizes not
    so big (to avoid overshooting). But what is the biggest possible step we can
    take in our gradient update while still keeping the training stable? That's the
    question that PPO tries to answer.
  </p>
</section>

<section>
  <p>
    One problem vanilla policy gradients have is that the performance of the
    agent increases over time (which in of itself is a good thing) but then - at
    some point - dramatically crashes and never recovers. That often leads to a
    reward curve like this:
  </p>
  <Figure
    path="Figure_1.png"
    caption="Hypothetical - yet realistic - scenario"
  />
  <p>
    This is very frustrating! Your agent was so close to perfection and then it
    plummets into the abyss forever. What a tragedy :( We want to add some
    constraints to ensure that our agent doesn't experience this kind of
    catastrophic forgetting.
  </p>
</section>
<section>
  <p>
    Let's change the objective function a bit and remove the total trajectory
    return <Katex math={"R(\\tau)"} /> from the equation. Instead, we will use the
    <i>advantage function</i>
    <Katex math={"A(s_t, a_t)"} />. The advantage function gives us a measure of
    how good taking some action <Katex math={"a_t"} /> at state <Katex
      math={"s_t"}
    /> was compared to the expected return. More specifically, one way of defining
    it is <Katex math={`A(s_t, a_t) = Q(a_t, s_t) - V(s_t)`} displayMode />
    In other words, it's the difference between taking action <Katex
      math={"a_t"}
    /> at state <Katex math={"s_t"} /> and then following the policy and the value
    of being in state <Katex math={"s_t"} />. If you subtract those two, you're
    left with the value of performing action <Katex math={"a_t"} />. Note, that
    there other ways to calculate the advantage (and we will learn about one of
    them later). Also note that <Katex math={"Q"} /> and <Katex math={"V"} /> are
    themselves expectations and what we are comparing are expected values.
  </p>
</section>
<section>
  <p>
    Now, changing our objective function to include the advantage function, we
    get
    <Katex
      math={`\\nabla_{\\theta} J(\\pi_{\\theta}) = \\underset{\\tau \\sim \\pi_{\\theta}}{E}{\\sum_{t=0}^{T} \\nabla_{\\theta} \\log \\pi_{\\theta}(a_t |s_t) A(s_t, a_t)}`}
      displayMode={true}
    />
  </p>
  <p>
    But why do we bother with the advantage function at all? Can't we just use
    the return of the trajectory <Katex math={"R(\\tau)"} /> as we had before? Of
    course, we <i>could</i>, but the problem with it is that it as a lot of
    <b>variance</b>. Consider the following example in which your agent gets
    these rewards
  </p>
  <pre><code class="language-python">{`5, 0, -5, 10, 0, -5, -5`}</code></pre>
  <p>
    The sum of these rewards is 0, i.e. <Katex math={"R(\\tau) = 0"} />. This
    gives us very little information about what actions lead to which results.
    We have to look at each action within the trajectory (kind of like zooming
    into the trajectory) to get more information out of it.
  </p>
  <p>The advantage function also gives us another, very useful hint.</p>
</section>
