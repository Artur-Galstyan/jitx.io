---
    layout: ../../layouts/blogpost.astro
    title: Improving Policy Gradients 
    pubDate: 2023-11-26
    description: "In our other blog post, we implemented the simplest possible policy gradients algorithm and this time, we will improve our baseline implementation!"
    tags: ["reinforcement learning", "deep learning", "policy gradients"]
---

## Contents

## Introduction

In the last [blog post](/posts/reinforce-policy-gradient), we
implemented a simple policy gradient algorithm, and in this post, we will
add a few more bells and whistles to the algorithm to make it more
performant. It's recommended to read the previous post before continuing as
well as the [](/posts/rl-basics) post.

## Smarter Rewards

In our current implementation, we used the rewards of the whole episode to
update the policy. But at each timestep, we should much rather look at what
rewards we received **after** executing a certain action. It doesn't
matter what happened **before** we took this action or what kind of rewards
we had received so far. Only what happens afterwards matters. Let's say, we had
these rewards:

```
rewards = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

```

Let's say we took some action
$a$
at timestep 5. Then we should only look at the rewards from timestep 5 to 10
to update our policy and similarly for every other timestep too. This is called
the **reward-to-go** and can be implemented like this (in JAX fashion):

```python
def get_total_discounted_rewards(rewards: Float[Array, "n_steps"], gamma=0.99) -> Array:
"""Calculate the total discounted rewards for a given set of rewards.
Args:
rewards: The rewards to calculate the total discounted rewards for.
gamma: The discount factor.
Returns:
The total discounted rewards.
"""
    def scan_fn(carry, current_reward):
        discounted_reward = carry + current_reward
        return discounted_reward * gamma, discounted_reward

    _, total_discounted_rewards = jax.lax.scan(scan_fn, 0.0, rewards[::-1])

    total_discounted_rewards = total_discounted_rewards[::-1].reshape(
        -1,
    )
    assert (
        total_discounted_rewards.shape == rewards.shape
    ), f"total_discounted_rewards.shape: {total_discounted_rewards.shape}, rewards.shape: {rewards.shape}"

    return total_discounted_rewards

if __name__ == "__main__":
    from icecream import ic
    r = jnp.array([1 for _ in range(10)])
    ic(get_total_discounted_rewards(r))
```

```
ic| get_total_discounted_rewards(r): Array([9.561792 , 8.648275 , 7.725531 , 6.7934656, 5.8519855, 4.9009953,
3.940399 , 2.9701 , 1.99 , 1. ], dtype=float32, weak_type=True)
```

## Reducing Variance with Baselines

Rewards in RL can be very noisy and can have a high variance, which can be
problematic for us. We're trying to estimate the policy gradient directly by
sampling from the environment, and if the rewards are very noisy, we get
unstable gradients. To reduce the variance, we can subtract a baseline from
the rewards. We can do that, because the baseline function with only take
the state as input and does not depend on our parameters
$\theta$. This means that the baseline term will effectively be a constant and will not
affect the gradient:

$$
\underset{a_t \sim \pi_{\theta}}{E} \left[{\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)}\right] = 0
$$

This means we can rewrite the policy gradient as:

$$
\nabla_{\theta} J(\pi_{\theta}) = \underset{\tau \sim \pi_{\theta}}{E}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\right)}
$$

The second part, i.e. $\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)$, is the "reward-to-go" part, we discussed in the previous section but with a
baseline subtracted from it.

Okay, but what to use as a baseline? Commonly, the value function is used as
a baseline, which also makes intuitive sense. Let's look at our reward
example from the previous section again:

```
ic| get_total_discounted_rewards(r): Array([9.561792 , 8.648275 , 7.725531 , 6.7934656, 5.8519855, 4.9009953,
                                            3.940399 , 2.9701   , 1.99     , 1.       ],      dtype=float32, weak_type=True)
```

Our value function represents how _good_ it is to be in a particular
state. Now, let's assume we've trained our value function a bit already (the
value function will be estimated using a neural network). And at timestep 0,
our value function estimates a value of $7$. We subtract from our reward
$9.561792$ the value function estimate $7$ and get $2.561792$.
This means that the action we took at timestep 0 was better than expected!
Remember, the value function expected a reward of $7$
but we got $9.561792$ . Our agent is positively surprised and will update its policy accordingly to
ensure we take that action in that state more often. This also goes the other
way around, too.

To implement this baseline, we will introduce another neural network, like
this one:

```python
class ValueNetwork(eqx.Module):
"""Value network for the policy gradient algorithm in a discrete action space."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        width_size: int = 64,
        depth: int = 3,
    ) -> None:
        key, *subkeys = jax.random.split(key, 5)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, x: Float32[Array, "state_dims"]) -> Array:
        """Forward pass of the policy network.
        Args:
            x: The input to the policy network.
        Returns:
            The output of the policy network.
        """
        return self.mlp(x)
```

This network is trained separately from the policy network using a different
loss function. In fact, its loss function is much simpler than the one from
the policy network because what we have here is a simple **supervised learning**

problem, where the target is the actual reward we got. In other words, the
loss function is:

```python
def value_loss_fn(
    value_network: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    rewards: Float32[Array, "n_steps"],
    ) -> Array:
"""Calculate the value loss for a given set of states and rewards.
Args:
value_network: The value network.
states: The states to calculate the value loss for.
rewards: The rewards to calculate the value loss for.
Returns:
The value loss.
"""
values = eqx.filter_vmap(value_network)(states)
return jnp.mean((values - rewards) \*\* 2)

@eqx.filter*jit
def step_value_network(
    value_network: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    rewards: Float32[Array, "n_steps"],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    ):
*, grad = eqx.filter_value_and_grad(value_loss_fn)(value_network, states, rewards)
updates, optimiser_state = optimiser.update(grad, optimiser_state, value_network)
value_network = eqx.apply_updates(value_network, updates)

return value_network, optimiser_state
```

This is simply the mean squared error loss. We will have to update our step
and loss function for our policy network:

```python
def objective_fn(
    policy: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    value_network: PyTree,
    ):
logits = eqx.filter_vmap(policy)(states)
log_probs = jax.nn.log_softmax(logits)
log_probs_actions = jnp.take_along_axis(
    log_probs, jnp.expand_dims(actions, -1), axis=1
)
rewards = rl_helpers.get_total_discounted_rewards(
    rewards
) # don't let the past distract you!
values = eqx.filter_vmap(value_network)(states)
advantages = rewards - values
return -jnp.mean(log_probs_actions \* advantages)

@eqx.filter*jit
def step(
    policy: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    value_network: PyTree,
    ):
*, grad = eqx.filter_value_and_grad(objective_fn)(
    policy, states, actions, rewards, value_network
)
updates, optimiser_state = optimiser.update(grad, optimiser_state, policy)
policy = eqx.apply_updates(policy, updates)

return policy, optimiser_state

```
