---
    layout: ../../layouts/blogpost.astro
    title: Monte Carlo Tree Search
    pubDate: 2024-06-23
    description: "Monte-Carlo Tree Search is the algorithm behind the success
    of DeepMind's AlphaGo and in this blog post, we will learn all about it!"
    tags: ["mcts", "mctx"]
---

# Monte Carlo Tree Search

(Draft)

2024-06-23

## Contents

## Introduction

Monte Carlo Tree Search (MCTS) is a heuristic search algorithm to find the best decision by iteratively running simulations of the process (e.g. a chess game). In fact, MCTS was effectively used in AlphaGo to solve the game of Go, giving good results even if the underlying domain becomes exceptionally large.

To implement MCTS, we first need to know how it works. MCTS consists of 4 steps:

- Traversal
- Expansion
- Rollout
- Backpropagation

So, let's go through each step.

### Traversal

In traversal, we first need to find a so-called _leaf node_. A leaf node is a node in a tree, which has no children. To do that, we apply the following simple algorithm:


![Traversal](/posts/monte-carlo-tree-search/MCTS_Traversal.drawio.svg)

We can very easily check if a node is a leaf node, but what is the `select_child_node` function? This function decides, how we select the next child node to traverse through. Once we reach a leaf node, we will use that to either expand the tree or perform a rollout. More on that later.

So, how do we choose this function? Well, it should be a function which takes in a list of nodes and returns the one it deems most appropriate. Here's a list of functions we could use for that (which I got from Claude):

- Upper Confidence Bound 1 (UCB1):
  Provides a good mix for exploration and exploitation
- Epsilon-Greedy:
  A simple approach where the best-known action is chosen most of the time, but with a small probability ε, a random action is selected.
- Thompson Sampling:
  This method uses Bayesian inference to select actions based on their probability of being optimal.
- UCB1-Tuned:
  An improved version of UCB1 that takes into account the variance of the rewards.
- EXP3 (Exponential-weight algorithm for Exploration and Exploitation):
  Designed for adversarial bandit problems, it's more robust in non-stationary environments.
- PUCB (Predictor + UCB):
  Combines UCB with a learned predictor to guide exploration.
- UCB-V:
  A variant that takes into account both the empirical variance and the number of samples.
- KL-UCB (Kullback-Leibler UCB):
  Based on information theory, it can be more efficient than UCB1 in some scenarios.
- Softmax Selection:
  Chooses actions probabilistically based on their estimated values.
- UCT (Upper Confidence bounds applied to Trees):
  A variation of UCB1 specifically adapted for tree search.
- RAVE (Rapid Action Value Estimation):
  An enhancement that shares value estimates between similar actions in different states.

In other words, you give it a list of child nodes, the function calculates (e.g.) the UCB1 value for each child node and then you pick the child node, which maximises the UCB1 score. Let's start with UCB1 and choose this as our function of choice. Later, we will simply let the user either pick a predefined function or provide his own function.

The mathematics of UCB1 are quite simple:

$$
  UCB1(s) = V(s) + C \sqrt{\frac{\ln n(s_{parent})}{n(s)}},
$$
where $V(s)$ is the average value of the state $s$, $C$ is a constant to balance exploration and exploitation, $n(s_{parent})$ is the number of visits to the **parent** node of $s$ and $n(s)$ is the number of visits to the state $s$.

In Python, we implement this as follows:

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def ucb1(
    avg_node_value: Float[Array, ""],
    visits_parent: Float[Array, ""],
    visits_node: Float[Array, ""],
    exploration_exploitation_factor: Float[Array, ""] = jnp.array(2.0),
) -> Float[Array, ""]:
    """
    Upper Confidence Bound 1 (UCB1) formula for MCTS.

    Args:
        avg_node_value: The average value of the current node. V(s)
        visits_parent: The number of visits of the parent node. N(s_parent)
        visits_node: The number of visits of the current node. n(s)
        exploration_exploitation_factor: The exploration-exploitation factor
            that balances between exploration and exploitation. C

    Returns:
        The UCB1 value of the current node. UCB1(s)
    """
    return avg_node_value + exploration_exploitation_factor * jnp.sqrt(
        jnp.log(visits_parent) / visits_node
    )
```

Simple stuff. Now, how do we go about traversing the tree? Where is the tree anyway? What does the data structure implementation look like?

Well, the answer is that most of the time, there won't be a static tree sitting somewhere to be passed around. In reality, we will create *callbacks* which construct the tree as we step forward in our environment.

In other words, we need to construct 2 functions: one that creates the root node (analogous to the `env.reset()` function if you're familiar with the `gym` API) and a kind of `env.step()` function (which we will just call `step`). Imagine you'd have to write out a tree for the entire game of chess. That's impossible which is why you simply create those 2 functions which determine the environment dynamics.

Let's say you have a neural network or something and that thing has a bunch of parameters, which you would want to *give* the MCTS algorithm because you want the `step` function of MCTS to have access to your neural network parameters (in order to take the right actions). The `step` function of the MCTS takes your parameters (among other things) as input and then applies the environment dynamics. Then, for each action you compute the UCB1, choose the next action and repeat the process.

Since we're building a library, we want to be as explicit as we can with our types, otherwise using Python libraries that don't force you into a certain kind of API can be a bit *wishy washy*.

For that, we will declare 2 callables: a `RootFnCallable` and the step function `StepFnCallable`. Now, we need to think about what we want from the user. We need from the user the parameters of whatever algorithm is being used, the current average value of the root node as well as the state representation of the root node. Let's define a struct which holds those values and force the user to return us this object.

```python
class RootFnOutput:
    params: PyTree
    value: Array
    state: PyTree


RootFnCallable = Callable[..., RootFnOutput]
```

Let's say we did something like this:

```python
def get_root() -> int:
    return 123


def forward(root_fn: RootFnCallable):
    root_fn()


forward(get_root)
```

Then Pyright should complain here saying something like:

```
Argument of type "() -> int" cannot be assigned to parameter "root_fn" of type "RootFnCallable" in function "forward"
  Type "() -> int" is incompatible with type "RootFnCallable"
    Function return type "int" is incompatible with type "RootFnOutput"
      "int" is incompatible with "RootFnOutput"
```

Which means we can now force the user to follow our API.
