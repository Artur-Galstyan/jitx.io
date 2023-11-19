<script lang="ts">
    import Katex from "$lib/components/Katex.svelte";
    import HintBox from "$lib/components/HintBox.svelte";
</script>

<section>
    <h3>Introduction</h3>
    <p>
        Before we dive deep into the REINFORCE algorithm, I <b>highly</b> suggest you first read the <a
            href="/posts/rl-basics">basics of RL</a> before continuing. This post will assume that you have a basic
        understand of the concepts introduced in the basics of RL post. Some parts will be repeated here for
        completeness, but it is highly recommended to read the basics of RL post first.
    </p>
</section>
<section>
    <h3>
        The Objective Function
    </h3>
    <p>
        We know that there are two ways to define the return from a <i>trajectory</i>:
    </p>
    <ul>
        <li>
            finite horizon undiscounted return
            <Katex math={`R(\\tau) = \\sum_{t=0}^T r_t`} displayMode={true}/>
        </li>
        <li>
            infinite horizon discounted return

            <Katex math={`R(\\tau) = \\sum_{t=0}^\\infty \\gamma^t r_t`} displayMode={true}/>
        </li>
    </ul>
    <p>
        And we also know that the probability to <i>"sample"</i> a trajectory depends on the current policy and is
        defined as
        <Katex math={`P(\\tau | \\pi) = p_0(s_0) \\prod_{t=0}^{T-1} P(s_{t + 1} | a_t, s_t) \\pi_\\theta(a_t|s_t)`}
               displayMode={true}/>
        With that in mind, we can define the objective function as the expected return, in other words:

        <Katex math={`J(\\pi_\\theta) = \\underset{\\tau \\sim \\pi_\\theta}{\\mathbb{E}}\\left[R(\\tau)\\right] = \\int_T P(\\tau | \\pi_\\theta) R(\\tau)`}
               displayMode={true}/>
    </p>
</section>
<section>
    <h3>
        The Policy Gradient
    </h3>
    <p>
        Here's a helpful hint: think of gradients as an <i>operation</i>, like plus or minus. In the case of plus or
        minus, the operation is denoted as
        <Katex math={`+`}/>
        or
        <Katex math={`-`}/>
        but for the gradient, we
        denote that as
        <Katex math={`\\nabla_x`}/>
        , where
        <Katex math={`x`}/>
        is the parameter we want to take the gradient with respect to. In our case, it's
        <Katex math={`\\theta`}/>
        . So, if we take the gradient of the objective function like so:
        <Katex math={`\\nabla_\\theta \\left(J(\\pi_\\theta)\\right)`} displayMode={true}/>
        we get a vector which points in the direction of <i>steepest ascend</i>.
    </p>
    <HintBox
            content={`We actually don't need the paranthesis around J; it's just there to emphasise that the gradient operator
             applies only to the next term. Since there is only a single term, the paranthesis are redundant.`}></HintBox>
    <p>
        Let's understand the gradient a bit better and use a simple example function like
        <Katex math={`f(\\theta) = f(\\theta_0, \\theta_1) = {2\\theta_0}^3 + {3\\theta_1}^4`} displayMode={true}/>

        In RL notation, we usually just write
        <Katex math={`\\pi_\\theta`}/>
        as a shortform for
        <Katex math={`\\pi_{\\theta_0, \\theta_1, \\dots, \\theta_n}`}/>
        where each
        <Katex math={`\\theta_i`}/>
        is a parameter in the model (i.e. the neural network).
    </p>
    <p>
        When we compute the gradient of
        <Katex math={`f`}/>
        with respect to the parameters
        <Katex math={`\\theta`}/>
        , we get a vector as result! You might be wondering why the result is a vector. Think of it this way: for you,
        it should be no problem to compute the derivative of the function
        <Katex math={`f`}/>
        w.r.t.
        <Katex math={`\\theta_0`}/>
        or w.r.t.
        <Katex math={`\\theta_1`}/>
        individually. In fact, the results of both derivates are
        <Katex math={`\\frac{\\partial f}{\\partial \\theta_0} = {6\\theta_0}^2`} displayMode={true}/>
        <Katex math={`\\frac{\\partial f}{\\partial \\theta_1} = {12\\theta_1}^3`} displayMode={true}/>
    </p>
    <p>
        A gradient simply computes <i>all</i> derivates w.r.t. every parameter at the same time. That's why you get a
        vector; it's because the function has multiple parameters and you need a vector that's as long as the number
        of parameters of your function (in our simple example, that would be 2). So, the gradient is:

        <Katex math={`\\nabla_\\theta f = \\begin{bmatrix} 6\\theta_0^2 \\\\ 12\\theta_1^3 \\end{bmatrix}`}
               displayMode={true}/>
    </p>
    <p>
        Now you have a vector which has the same size as the parameters. This gradient points to the direction of
        steepest ascend, which means that if you add this vector to
        <Katex math={`\\theta`}/>
        , then the underlying function
        <Katex math={`f`}/>
        will <i>increase</i>! Similarly, if you subtract the gradient, the underlying function will <i>decrease</i>.
        That's why - in the context of machine learning - people talk about gradient descent. You take the gradient of
        the loss function and subtract it from your parameters, because you want to make the loss smaller. In our case,
        it's reverted because we have an objective function which we want to make as big as possible (because that means
        we got more reward). This also means, that we take the gradient and <i>add</i> it to our parameters as that
        makes
        <Katex math={`J`}/>
        bigger.
    </p>
    <p>
        So back to RL, we want to find the gradient of
        <Katex math={`J`}/>
        so that we can add the gradient vector to our parameters
        <Katex math={`\\theta`}/>
        to get more reward:
        <Katex math={`\\theta_{k+1} = \\theta_k + \\alpha \\nabla_{\\theta} J(\\pi_{\\theta}) |_{\\theta_k}
`} displayMode={true}/>
        Here,
        <Katex math={`\\alpha`}/>
        is the learning rate and
        <Katex math={`\\nabla_{\\theta} J(\\pi_{\\theta}) |_{\\theta_k}`}/>
        refers to the gradient w.r.t.
        <Katex math={`\\theta_k`}/>
        where
        <Katex math={`k`}/>
        is the current iteration. Simply put, take the current parameters, add the gradient on top, and voilá, you got
        the next set of parameters.
    </p>

</section>