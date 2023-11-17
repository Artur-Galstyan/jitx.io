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
        <Katex math={`\\tau \\sim \\pi_{\\theta}`} displayMode={true}/>
    </p>
    <p>
        Let's say we have a policy
        <Katex math={`\\pi`}/>
        that we want to evaluate. Since what we have here is essentially a stochastic process, we can simply evaluate
        the policy by sampling a large number of trajectories from the environment and then averaging the rewards of
        these trajectories. We can then define the objective as the expected return over all trajectories:
    </p>
    <Katex math={`J(\\pi_{\\theta}) = \\mathbb{E}_{\\tau \\sim \\pi_{\\theta}} \\left[ R_t(\\tau) \\right] = \\mathbb{E}_{\\tau \\sim \\pi_{\\theta}} \\left[ \\sum_{i=0}^{T} \\gamma^i r_i \\right]
`} displayMode={true}/>
    <p>
        Given the objective function, the goal can be stated as finding the set of parameters for the policy that
        maximize the objective function, i.e. get's the highest expected reward:
    </p>
    <Katex math={`\\max_{\\theta} J(\\pi_{\\theta}) = \\mathbb{E}_{\\tau \\sim \\pi_{\\theta}} \\left[ R(\\tau) \\right]
`} displayMode={true}/>
</section>
<section>
    <h3>The Policy Gradient Theorem</h3>
    <p>
        Above, we have defined the objective function and the goal is to find the parameters
        <Katex math={`\\theta`}/>
        that maximize the objective function. In order to do that, we need to compute the gradient of the objective
        function with respect to the parameters
        <Katex math={`\\theta`}/>
        . Remember, gradients are vectors, which point in the direction of steepest ascent. If you had the gradient,
        you could add the gradient to the parameters and this would make the objective function larger. Let's write the
        gradient of the objective function as
    </p>
    <Katex math={`\\nabla_{\\theta} J(\\pi_{\\theta}) = \\nabla_{\\theta} \\mathbb{E}_{\\tau \\sim \\pi_{\\theta}} \\left[ R(\\tau) \\right]
`} displayMode={true}/>
    <p>
        This is where it gets a little tricky. The problem is that
        <Katex math={`R(\\tau)`}/>
        is an unknown function and it has no parameter
        <Katex math={`\\theta`}/>
        , which means that we can't compute the gradient of the objective function with respect to
        <Katex math={`\\theta`}/>
        . This means that we have to rewrite the gradient of the objective function such a way as to make it rely on
        the things we can influence, i.e. the policy parameters
        <Katex math={`\\theta`}/>
        .
    </p>
    <p>
        Let's have a look at the most general case. Essentially, what we have here is a function
        <Katex math={`f(x)`}/>
        (which stands for the reward function
        <Katex math={`R(\\tau)`}/>
        )
        and a parameterized probability distribution
        <Katex math={`p(x|\\theta)`}/>
        (which stands for the policy
        <Katex math={`\\pi(\\tau|\\theta)`}/>
        ). With that we can rewrite
        <Katex math={`\\mathbb{E}_{x \\sim p(x|\\theta)}[f(x)]`}/>
        as the following transformations:
    </p>
    <p>
        <i>Definition of Expectation:
        </i>
        <Katex math={`\\nabla_{\\theta} \\mathbb{E}_{x \\sim p(x|\\theta)}[f(x)] = \\nabla_{\\theta} \\int f(x) p(x|\\theta) dx
`} displayMode={true}/>
        <i>Bring in the gradient:</i>
        <Katex math={`= \\int \\nabla_{\\theta} (f(x) p(x|\\theta)) dx
`} displayMode={true}/>
        <i>Apply chain rule:</i>
        <Katex math={`= \\int \\left( f(x) \\nabla_{\\theta} p(x|\\theta) + \\nabla_{\\theta} f(x) p(x|\\theta) \\right) dx
`} displayMode={true}/>
        <i>
            <Katex math={`\\nabla_\\theta f(x) = 0`}/>
            because
            <Katex math={`f(x)`}/>
            is independent of
            <Katex math={`\\theta`}/>
        </i>
        <Katex math={`= \\int f(x) \\nabla_{\\theta} p(x|\\theta) dx
`} displayMode={true}/>
        <i>Multiply
            <Katex math={`\\frac{p(x|\\theta)}{p(x|\\theta)}`}/>
        </i>
        <Katex math={`= \\int f(x) \\frac{\\nabla_{\\theta} p(x|\\theta)}{p(x|\\theta)} p(x|\\theta) dx
`} displayMode={true}/>
        <i>Log trick
            <Katex math={`\\frac{\\nabla_{\\theta} p(x|\\theta)}{p(x|\\theta)} = \\nabla_{\\theta} \\log p(x|\\theta)
`}/>
        </i>
        <Katex math={`= \\int f(x) p(x|\\theta) \\nabla_{\\theta} \\log p(x|\\theta) dx
`} displayMode={true}/>
        <i>Definition of Expectation</i>
        <Katex math={`= \\mathbb{E}_{x \\sim p(x|\\theta)} \\left[ f(x) \\nabla_{\\theta} \\log p(x|\\theta) \\right]
`} displayMode={true}/>
        <Katex math={`\\Rightarrow \\nabla_{\\theta} J(\\pi_{\\theta}) = \\mathbb{E}_{\\tau \\sim \\pi_{\\theta}} \\left[ R(\\tau) \\nabla_{\\theta} \\log \\pi(\\tau|\\theta) \\right]
`} displayMode={true}/>
    </p>
    <p>
        The only thing missing here is that we don't have
        <Katex math={`p(\\tau|\\theta)`}/>
        . This means, we have to apply some set of transformations to get from
        <Katex math={`\\pi(\\tau|\\theta)`}/>
        to something we can control and/or have access to. Notice that
        <Katex math={`p(\\tau|\\theta)`}/>
        is the probability to get a trajectory
        <Katex math={`\\tau`}/>
        given the policy
        <Katex math={`\\pi(\\tau|\\theta)`}/>
        . We can write this as a product of the probability to transition from state
        <Katex math={`s_t`}/>
        to state
        <Katex math={`s_{t+1}`}/>
        given the action
        <Katex math={`a_t`}/>
        and the probability to take that action under our current policy.
    </p>
    <Katex math={`p(\\tau | \\theta) = \\prod_{t \\geq 0} p(s_{t+1} | s_t, a_t) \\pi_{\\theta}(a_t | s_t)
`} displayMode={true}/>
    <p>
        If we apply logs to both sides, we get the following:
    </p>
    <i>This first line defines the probability of a trajectory
        <Katex math={`\\tau`}/>
        , given the policy parameters
        <Katex math={`\\theta`}/>
        , as the product of the state transition probabilities and the policy's action selection probabilities.</i>
    <Katex math={`\\log p(\\tau | \\theta) = \\sum_{t \\geq 0} \\log p(s_{t+1} | s_t, a_t) + \\log \\pi_{\\theta}(a_t | s_t)
`} displayMode={true}/>
    <i>Taking the logarithm of the trajectory probability gives a sum of the logarithms of the state transition
        probabilities and the action probabilities.</i>
    <Katex math={`\\nabla_{\\theta} \\log p(\\tau | \\theta) = \\nabla_{\\theta} \\sum_{t \\geq 0} \\log p(s_{t+1} | s_t, a_t) + \\log \\pi_{\\theta}(a_t | s_t)
`} displayMode={true}/>
    <i>The gradient of the log probability of the trajectory with respect to
        <Katex math={`\\theta`}/>
        is the sum of the gradients of the log probabilities.</i>
    <Katex math={`\\nabla_{\\theta} \\log p(\\tau | \\theta) = \\sum_{t \\geq 0} \\nabla_{\\theta} \\log p(s_{t+1} | s_t, a_t) + \\nabla_{\\theta} \\log \\pi_{\\theta}(a_t | s_t)
`} displayMode={true}/>
    <i>Since the state transition probabilities
        <Katex math={`p(s_{t+1} | s_t, a_t)`}/>
        are independent of
        <Katex math={`\\theta`}/>
        , their gradients are zero, which simplifies the expression.
    </i>
    <Katex math={`\\nabla_{\\theta} \\log p(\\tau | \\theta) = \\sum_{t \\geq 0} \\nabla_{\\theta} \\log \\pi_{\\theta}(a_t | s_t)
`} displayMode={true}/>
    <i>Only the gradients of the log policy probabilities with respect to
        <Katex math={`\\theta`}/>
        remain</i>

</section>