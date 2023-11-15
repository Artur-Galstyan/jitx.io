<script lang="ts">
    import Figure from "$lib/components/Figure.svelte";
    import Katex from "$lib/components/Katex.svelte";
</script>

<section>
    <h3>Introduction</h3>
    <p>
        Reinforcement learning (RL) is a subfield of machine learning in which an agent tries to maximize its reward by
        interacting with its environment.
    </p>
    <Figure path="rl-setting.webp"></Figure>
    <p>
        This makes RL already very different from other machine learning subfields like supervised learning (SL) or
        unsupervised learning (UL). This is because of the interactive nature of RL. While in SL or UL, the data is
        given to the learning algorithm apriori, in RL, the agent has to generate it from scratch.
    </p>
    <p>
        RL algorithms can be categorised based on the way they solve the environment:
    </p>
    <ul>
        <li>
            Model-based RL: Solves the environment by learning its dynamics (e.g. by predicting the next state)
        </li>
        <li>
            Model-free RL: Solves the environment by learning the optimal policy/value function directly
        </li>
    </ul>
    <p>
        The algorithm for this post is a model-free, policy-based RL algorithm. We will explore the other algorithms in
        future posts.
    </p>
</section>
<section>
    <h3>
        The Objective Function
    </h3>
    <p>
        As mentioned in the beginning, the goal of RL is to maximize the reward. Let's define a trajectory
        <Katex math={`\\tau`}/>
        as a sequence of tuples
        <Katex math={`(s_t, a_t, r_t)`}/>
        in other words
        <Katex math={`\\tau \\approx \\left\\{ (s_0, a_0, r_0), (s_1, a_1, r_1), \\ldots, (s_T, a_T, r_T) \\right\\}
`} displayMode/>
    </p>
    <p>
        where
        <Katex math={`s_t`}/>
        is the state at time
        <Katex math={`t`}/>
        ,
        <Katex math={`a_t`}/>
        is the action taken at time
        <Katex math={`t`}/>
        , and
        <Katex math={`r_t`}/>
        is the reward received at time
        and
        <Katex math={`T`}/>
        is the final time step. Now, we can define the reward in terms of the trajectory as simply the sum of all the
        rewards received from timestep
        <Katex math={`t`}/>
        to
        <Katex math={`T`}/>
        :
    </p>
    <Katex math={`R_t(\\tau) = \\sum_{i=t}^{T} r_i
`} displayMode/>
    <p>
        However, this is not exactly what we <i>should want</i>. The equation above indicates, that all rewards are
        equally weighted and therefore equally important. But think for a moment that you are the agent and the
        environment is a game of chess.
    </p>
    <p>
        The reason you weigh rewards differently is that the further you plan ahead, the more uncertain your predictions
        become. In that case, does it really make sense to weigh the rewards you might get 10 steps from now equally to
        the rewards you get in the next step, if the probability of reaching the desired state 10 steps from now is only
        1%? Similarly, you could win the game in the next move and get a garanteed reward right now or play a move that
        might lead to a win in 10 moves, but also might lead to a loss in 10 moves. In that case, it would make sense to
        give the immediate reward a higher weight than the reward you might get in 10 moves. <span
            class="font-bold text-accent">In other words, the further
        away the reward is, the less it should be weighted</span>. In RL, weighing the rewards is called discounting.
        Our
        reward equation then becomes:
    </p>
    <Katex math={`R_t(\\tau) = \\sum_{i=t}^{T} \\gamma^{i-t} r_i`} displayMode={true}/>
    <p>
        For brevity, we denote the case where
        <Katex math={`t=0`}/>
        as simply
        <Katex math={`R(\\tau)`}/>
        .
    </p>
    Notice, that the discount factor
    <Katex math={`\\gamma`}/>
    is a number between 0 and 1, and, that the discounting starts with should the exponent
    0, i.e.
    <Katex math={`\\gamma^0 = 1`}/>
    , which is why we have subtracted
    <Katex math={`t`}/>
    from
    <Katex math={`i`}/>
    .
    Take
    <Katex math={`t=3`}/>
    and
    <Katex math={`T=5`}/>
    for example. The sequence of discount factors would be
    <Katex math={`\\gamma^0, \\gamma^1, \\gamma^2, \\gamma^3, \\gamma^4, \\gamma^5`}/>
    and if we didn't subtract
    <Katex math={`t`}/>
    from
    <Katex math={`i`}/>
    , we would get
    <Katex math={`\\gamma^3, \\gamma^4, \\gamma^5, \\gamma^6, \\gamma^7, \\gamma^8`}/>
    instead. This would mean that the reward for the first step would be discounted by
    <Katex math={`\\gamma^3`}/>
    instead of
    <Katex math={`\\gamma^0`}/>
    . This is not what we want. It makes no sense for the agent to discount the immediate next reward by a factor of
    <Katex math={`\\gamma^3`}/>
    , as it is it's first move after timestep
    <Katex math={`t`}/>
    .
    <p>
        These trajectories are also called episodes and we will use these terms interchangeably. Trajectories or
        episodes are entities that can be sampled from the environment by interacting with it through the agent's
        policy. Mathematically, we can write the sampling process as:
        <Katex math={`\\tau \\approx \\pi_{\\theta}`} displayMode={true}/>
    </p>
</section>