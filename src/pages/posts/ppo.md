---
    layout: ../../layouts/blogpost.astro
    title: Proximal Policy Optimization
    pubDate: 2024-01-02
    description: "PPO is one of the most successful RL algorithms in existance and in this blog post you will learn all about it!"
    tags: ["reinforcement learning", "deep learning", "actor-critic"]
---

# Proximal Policy Optimization

2024-01-02

## Contents

## Introduction

Proximal Policy Optimization (PPO) is one of the most successful RL algorithms out there. It is known for its ease of tuning and excellent performance. Today, we will cover PPO and you will see how to implement it yourself using Jax! So, let's get started!

First, where does PPO fall into the RL algorithm buckets? PPO is an Actor-Critic algorithm, which means it falls into the policy gradient faction. In fact, it's a natural extension to the vanilla policy gradient algorithm as we will see in a moment. First, let's review the vanilla policy gradient algorithm (which we had already covered in this blogpost before).

First, we define the reward (infinite horizon and discounted) given some trajectory

$$
R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t


$$

Next, what is the probability to sample that particular trajectory? We define it as

$$
P(\tau|\pi) = p_0(s_0) \prod_{t=0}^{\infty} p(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t|s_t)
$$

Spoken out loud, the above equation translates to the probability to sample a particular trajectory $\tau$ given a policy $\pi$ with parameters $\theta$ is equal to the the probability of the initial state times the product of the probability to land in a state $s_{t+1}$ given the current state $s_t$ and the action $a_t$ times the probability to pick the action $a_t$ given the current state $s_t$ .

Now that we now the probability to sample a trajectory given our current policy, we define the objective function as

$$
J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} R(\tau) = \int P(\tau|\pi_{\theta}) R(\tau)
$$

Given that objective function, we next computed the gradient of the objective function. I won't go into the details here, but if you want to understand the details, I suggest you read my other blog post. Anyway, the gradient of the objective function is defined as

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ R(\tau) \sum_{t} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \right]
$$

Furthermore, we had also improved this objective function by including a baseline in the form of a value network, like so:

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \left( \sum_{t'=t}^{T} R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t) \right)
$$

With this gradient, we can update the parameters of our policy, which in turn should increase our rewards. In the update function, we had also included a learning rate αα to make our step sizes not so big (to avoid overshooting). But what is the biggest possible step we can take in our gradient update while still keeping the training stable? That's the question that PPO tries to answer.

One problem vanilla policy gradients have is that the performance of the agent increases over time (which in of itself is a good thing) but then - at some point - dramatically crashes and never recovers. That often leads to a reward curve like this:

[Reward Curve](/posts/ppo/Figure_1.png)

This is very frustrating! Your agent was so close to perfection and then it plummets into the abyss forever. What a tragedy :( We want to add some constraints to ensure that our agent doesn't experience this kind of catastrophic forgetting.

Let's change the objective function a bit and remove the total trajectory return $R(\tau)$ from the equation. Instead, we will use the advantage function $A(s_t,a_t)$. The advantage function gives us a measure of how good taking some action $a_t$ at state $s_t$ was compared to the expected return. More specifically, one way of defining it is

$$
A(s_t,a_t)=Q(a_t,s_t)-V(s_t)
$$

In other words, it's the difference between taking action $a_t$ at state $s_t$ and then following the policy and the value of being in state $s_t$. If you subtract these two, you get left with the value of performing action $a_t$.

Note, that there are other ways to calculate the advantage function (and we will learn about one of them later). Also note that $Q$ and $V$ are themselves expectations and what we are comparing are expected values.

Now, changing our objective function to include the advantage function, we get

$$
\nabla_\theta J(\pi_\theta)=E_{\tau\sim\pi_\theta}\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)A(s_t,a_t)
$$

But why do we bother with the advantage function at all? Can't we just use the return of the trajectory $R(\tau)$ as we had before? Of course, we could, but the problem with it is that it as a lot of variance. Consider the following example in which your agent gets these rewards:

```
5, 0, -5, 10, 0, -5, -5
```

The sum of these rewards is 0, i.e. $R(\tau) = 0$. This gives us very little
information about what actions lead to which results. We have to look at
each action within the trajectory (kind of like zooming into the trajectory)
to get more information out of it.

The advantage function also gives us another, very useful hint. It's sign
tells us if we should pursue that action again in the future. If it's positive,
then the performed action has given as _more reward than expected_.
Remember, the advantage is the difference between $Q(s, a)$ and $V(s)$.
If it was 0, then the performed action at that state and the value of that
state were equal, i.e. exactly as expected. If it was negative, it would mean
that on average the agent was better off with any other action. Vice versa,
a positive advantage tells us that the performed action was better than
expected.

Let's put this information safely away in a drawer in our brain and come
back to PPO and the problem it's trying to solve, which was _mitigating
catastrophic forgetting_. At some point - in vanilla policy gradients - an
update happens which shouldn't have happened. PPO tries to avoid this
by making smaller updates.

The way PPO does is this by clipping the probability ratio within the
objective function into a certain range, which is the hyperparameter $\epsilon$. It
is defined as

$$
J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} \min(r_t(t)A(·), \text{clip}(r_t(t)A(·), 1 - \epsilon, 1 + \epsilon)) \right],
$$

where $A(·)$ is just a shortform for $A(s_t, a_t)$. We will talk about what the
_probability ratio_ means.

## Lightning FAQ

The vanilla objective function (not the gradient, just the objective function) was defined as

$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} R(\tau) = \int P(\tau|\pi_\theta)R(\tau)$

but why did we bother deriving the gradient at all? Don't we have fancy autograd packages which do all that for us?

The answer is that - yes, we do have amazing autograd packages but even they can't changes the laws of the universe. The issue - namely why we couldn't just let the autograd rip and do everything for us - was that we didn't have $P(\tau|\pi_\theta)$ because that depends on the dynamics of the environment - something we don't have. That's the whole reason, we have to do all that math to get this into a differentiable form.

Luckily, that issue is not present in the PPO objective function, which means we don't have to derive it by hand. That's because everything in the function is either something we have or can easily compute. Nice! Quick FAQ over!

## Back to PPO

Let's talk about the components of the objective function and we will cover the easy things first. As mentioned before, $\epsilon$ is a hyperparameter, which is between 0 and 1. The min function is hopefully something I don't have to explain. The clip function does exactly what numpy.clip does. Next up is the ratio $r_\theta(t)$, something a bit more interesting.

Mathematically, the ratio is defined as "the ratio of probabilities between your current policy and the previous policy", i.e.

$$
r_\theta(t) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

Let's stop and think for a moment what this ratio tells us by interpreting it. If the action probability in the current policy is higher than in the previous iteration (i.e. the action is now more likely than before), then the ratio will be positive. If the advantage function is now also positive, then the whole thing becomes positive, which will encourage the action even more in the future. And vice versa.

This ratio is like a scaling factor, i.e. it weighs the advantage function. We constrain the ratio in the range of $1 \pm \epsilon$ (that's how much the ratio is clipped by). We do this clipping, to prevent a too-large update, e.g. if the advantage for some reason just explodes in the positive direction, which would add a lot of variance and, thus, destabilise the training process.

In essence, the ratio tells us how likely a probability is now compared to before. It becomes more likely if the action lead to a positive advantage. A positive advantage is achieved if the action performed better than expected by our baseline (and vice versa).

One side note here: when we are referring to the policy $\pi(a_t|s_t)$, we talk about a probability distribution of taking action $a_t$ at state $s_t$. The authors in the PPO paper defined the ratio the same way we did earlier. In practice when we implement this, we use log probabilities. We use logs here because probabilities can get very small and our networks don't learn so effectively when very small numbers are involved. When you log everything, those small numbers get larger, which introduce numerical stability. So, when you implement this objective function, then in my opinion, the authors could have written it like this:

$$
r_\theta(t) = e^{\log(\pi_\theta(·))-\log(\pi_{\theta old}(·))}
$$

where $\pi(·)$ stands for $\pi(a_t|s_t)$. Note, that this is 100% equivalent mathematically to simply computing the ratios directly. But in using logs, we introduce more stability. Unfortunately, these details don't often make it into the paper - much to my disappointment, because these kind of notes are incredibly valuable to beginners in the field.
