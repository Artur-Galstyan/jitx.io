<script lang="ts">
    import Katex from "$lib/components/Katex.svelte";
    import Figure from "$lib/components/Figure.svelte";

    type StateTuple = {
        state: number
        action: number
        reward: number
    }

    let currentState = 0;
    let currentReward = 0;
    let trajectory: StateTuple[] = []
</script>


<p>
    Reinforcement Learning is my favourite machine learning area and in this post, I will teach you
    about the basics of Reinforcement Learning (RL) and about the basic notation as well as other
    formulations in RL. Let's get started!
</p>

<p>
    First, the general RL setting. In RL, we have an agent that interacts with an environment. The goal of the
    agent is to get as much reward as possible. At each time step, the agent can perform an action and receive
    a reward for that action. The figure below illustrates this.
</p>
<Figure path="rl-setting.webp" caption="The general RL setting."/>
<p>
    When the agent interacts with the environment, it generates a so called <i>trajectory</i>. That trajectory
    is a list of <i>state, action, reward</i> tuples, i.e.:
</p>
<Katex math={`\\tau = \\big( (s_0, a_0, r_0), \\dots, (s_t, a_t, r_t), \\dots (s_T, a_T, r_T) \\big)`}
       displayMode={true}/>
<p>
    A trajectory starts from timestep
    <Katex math={`t=0`}/>
    until
    <Katex math={`T`}/>
    which is the terminal timestep
    (i.e. the last timestep of the episode).
</p>
<p>
    Given a trajectory, you can compute the <i>return</i> of the trajectory. <i>How much return did you get from that
    trajectory?</i> There are two types of returns you can compute: either the absolute return or the discounted return.
    The former is simply a sum of the rewards for every timestep:
    <Katex math={`R(\\tau) = \\sum_{t=0}^{T} = r_t`} displayMode={true}/>
    This is also called a <b>finite horizon undiscounted reward</b>. For the other case, you discount the return by some
    factor
    <Katex math={`\\gamma \\in [0, 1]`}/>
    and it's referred to as <b>infinite horizon discounted return</b>. It's defined as
    <Katex math={`R(\\tau) = \\sum_{t=0}^{\\infty} = \\gamma^t r_t`} displayMode={true}/>
</p>
<p>
    At each timestep, the agent sees the current state and decides what action to take. The action can either be
    deterministic or stochastic. If it's deterministic, its written as
    <Katex math={`a_t = \\mu_\\theta(s_t)`} displayMode={true}/>
    and if it's stochastic, then the action is sampled from a distribution, i.e.
    <Katex math={`a_t \\sim \\pi_\\theta(s_t)`} displayMode={true}/>

    The
    <Katex math={`\\theta`}/>
    is there to emphasise that those <i>policies</i>, whether deterministic or stochastic,
    usually are parameterised functions, such as neural networks and
    <Katex math={`\\theta`}/>
    represents their parameters.
</p>
<p>
    The action itself can either be discrete (e.g. <i>press button A</i>) or continious (e.g. <i>actuate motor by
    4.2</i>). If it's discrete, the <i>logits</i> given from the policy can be <i>softmaxed</i> to become probabilities
    and then simply sampled from. If the actions are continious, then the policy usually returns a Gaussian distribution
    (i.e. the mean and deviation). Then you simply use those to sample your action.
</p>
<p>
    Let's go back to our trajectory. What are the odds of getting a particular trajectory? It depends on the current
    policy of the agent of course, so let's write that down:
    <Katex math={`P(\\tau | \\pi_\\theta) = p_0(s_0) \\prod_{t=0}^{T-1} P(s_{t+1} | a_t, s_t) \\pi_\\theta(a_t | s_t)`}
           displayMode={true}/>
    The equation above can be read as <i>"the probability of getting trajectory
    <Katex math={`\\tau`}/>
    given
    <Katex math={`\\pi_\\theta`}/>
    equals the probability of the initial state
    <Katex math={`s_0`}/>
    times the <b>products</b> of the probability to reach state
    <Katex math={`s_{t+1}`}/>
    given action
    <Katex math={`a_t`}/>
    and state
    <Katex math={`s_t`}/>
    times the probability to pick action
    <Katex math={`a_t`}/>
    given the current state
    <Katex math={`s_t`}/>
    .
    "
</i>
    As a useful shorthand, we can write
    <Katex math={`s' ~ P(\\cdot | s, a)`}/>
    , which reads as <i>sampling the next state from the environment, given
    the current state and action.</i>
</p>
<p>
    With this, we can now define our objective function
    <Katex math={`J(\\pi_\\theta) = \\int_T P(\\tau | \\pi_\\theta) R(\\tau)`} displayMode={true}/>
</p>
<p>
    Let's have a look at an extremely simple, interactive example: you're the agent and you can choose between two
    doors.
</p>
<div class="border border-primary rounded p-4">
    <div class="flex justify-center my-2">
        <button
                on:click={() => {
                    currentReward = 0;
                    currentState = 0;
                    trajectory = [];
                }}
                class="btn btn-sm btn-outline">
            Reset
        </button>
    </div>
    <div class="flex justify-center space-x-4">
        <button
                class:btn-disabled={currentState === 10}
                on:click={() => {
                   trajectory = [...trajectory, {
                       reward: 1,
                       state: currentState,
                       action: 1
                   }]
                    currentState++;
                   currentReward++;
               }}
                class="btn btn-primary">
            Door 1
        </button>
        <button
                class:btn-disabled={currentState === 10}
                on:click={() => {
                   trajectory = [...trajectory, {
                       reward: -1,
                       state: currentState,
                       action: 2
                   }]
                    currentState++;
                    currentReward--;
                    console.log(trajectory.length)
               }}
                class="btn btn-secondary">
            Door 2
        </button>
    </div>
    <div class="flex space-x-4 justify-center">
        <div class="text-center">
            Current State = {currentState}
        </div>
        <div class="text-center text-accent">
            Total Reward = {currentReward}
        </div>
    </div>

    <div class="text-center">Trajectory (state, action, reward):</div>

    <div class="text-center">
        {#each {length: trajectory.length} as _, i}
            <div>
                ({trajectory[i].state}, {trajectory[i].action}, {trajectory[i].reward})
            </div>
        {/each}
    </div>
</div>
<p>
    Let's say, your policy is to <b>always</b> pick door 2. Go ahead and choose door 2 all the time. The probability in
    that case to get that trajectory is <i>100%</i> (unless you throw some randomness in there). The total reward you
    get is
    <Katex math={`-10`}/>
    . In other words,
    given your policy of always choosing door 2 and the reward structure as is then
    <Katex math={`J(\\pi_\\theta) = -10`}/>
    . Simple stuff. The goal of the agent is to maximise this objective; to find a policy which maximises
    <Katex math={`J`}/>
    . In our example, that policy is to always pick door 1. On the other hand, if you pick an action randomly, then the
    probability of getting a specific trajectory becomes 0.5. Use the interactive game from above to experiment what
    <Katex math={`J`}/>
    becomes if you pick an action (uniformly) randomly.
</p>
<p>
    The objective function is also called the <i>expected return</i>:
    <Katex math={`J(\\pi_\\theta) = \\int_T P(\\tau | \\pi_\\theta) R(\\tau) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\\Big[ R(\\tau) \\Big]`}
           displayMode={true}/>
    And we're trying to maximise the expected return! By the way, the best possible policy (and also the <i>value
    functions</i> - we'll get to those) is written with a star, i.e.
    <Katex math={`\\pi^*`}/>
</p>
<p>
    There are 2 more questions to answer:
</p>
<ul>
    <li>
        How <i>good</i> is it to be in state
        <Katex math={`s_t`}/>
        and then follow my current policy?
    </li>
    <li>
        How <i>good</i> is it to be in state
        <Katex math={`s_t`}/>
        , perform some arbitrary action
        <Katex math={`a_t`}/>
        and <b>then</b> follow my current policy?
    </li>
</ul>
<p>
    The former is defined as the <i>on-policy</i> value function
    <Katex math={`V^\\pi(s)`}/>
    :
    <Katex math={`V^{\\pi}(s) = \\mathbb{E}_{\\tau \\sim \\pi} [R(\\tau) | s_0 = s]
`} displayMode={true}/>
    It's called <i>on-policy</i> because the agent stays <i>on the policy</i>, i.e. it doesn't "leave" the policy; it
    stays there.
    The latter is called the <i>on-policy</i> action-value function
    <Katex math={`Q^\\pi(s, a)`}/>
    :
    <Katex math={`Q^{\\pi}(s, a) = \\mathbb{E}_{\\tau \\sim \\pi} [R(\\tau) | s_0 = s, a_0 = a]
`} displayMode={true}/>
    Like with the optimal policy, both of the value functions have an <i>optimal</i> function, i.e.
    <Katex math={`V^*(s) = \\max_{\\pi} \\mathbb{E}_{\\tau \\sim \\pi} [R(\\tau) | s_0 = s]
`} displayMode={true}/>
    ( which is the maximum expected return over all policies
    <Katex math={`\\pi`}/>
    , given the initial state
    <Katex math={`s`}/>
    ) <br><br>
    and
    <Katex math={`Q^*(s, a) = \\max_{\\pi} \\mathbb{E}_{\\tau \\sim \\pi} [R(\\tau) | s_0 = s, a_0 = a]
`} displayMode={true}/>
    (which is the maximum expected return for taking action
    <Katex math={`a`}/>
    in state

    <Katex math={`s`}/>
    and thereafter following the best policy
    <Katex math={`\\pi`}/>
    )
</p>
<p>
    The optimal action-value function is a bit more useful, because we can simply do this to get the optimal action:
    <Katex math={`a^*(s) = \\arg\\max_a Q^*(s, a).
`} displayMode={true}/>
</p>
<p>
    The last bit of RL basics are the Bellman equations. These show a kind-of recursive property of the value functions,
    where the value of a state is the immediate return plus the discounted value of the next state. In other words:
    <Katex math={`V^{\\pi}(s) = \\mathbb{E}_{a \\sim \\pi, s' \\sim P} \\left[ r(s, a) + \\gamma V^{\\pi}(s') \\right]
`} displayMode={true}/>
    <Katex math={`V^{*}(s) = \\mathbb{E}_{a \\sim \\pi, s' \\sim P} \\left[ r(s, a) + \\gamma V^{*}(s') \\right]
`} displayMode={true}/>
    And the same goes for the action-value function:
    <Katex math={`Q^{\\pi}(s, a) = \\mathbb{E}_{s' \\sim P} \\left[ r(s, a) + \\gamma \\mathbb{E}_{a' \\sim \\pi} \\left[ Q^{\\pi}(s', a') \\right] \\right],
`} displayMode={true}/>
    <Katex math={`Q^*(s, a) = \\mathbb{E}_{s' \\sim P} \\left[ r(s, a) + \\gamma \\max_{a'} Q^*(s', a') \\right],
`} displayMode={true}/>
</p>
<p>
    The last important part is the advantage function. It basically describes how good it is to pick action
    <Katex math={`a`}/>
    in state
    <Katex math={`s`}/>
    compared to the average action-value function (i.e. samping
    actions from the policy). It's defined as
    <Katex math={`A_{\\pi}(s, a) = Q_{\\pi}(s,a) - V_{\\pi}(s)`} displayMode={true}/>
</p>
<p>
    And those are the basics of RL. In the next blog post, we will go one step further and implement REINFORCE (a policy
    gradient algorithm) and actually watch some agents in action! Stay tuned!
</p>
