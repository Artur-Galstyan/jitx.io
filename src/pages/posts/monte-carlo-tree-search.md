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
- Rollout (*)
- Backpropagation

**Quick footnote on the rollout step**:
Newer versions of MCTS don't use the traditional rollout step anymore. In traditional MCTS, you would have a default policy (e.g. action randomly) and use that to step through the environment until you reach a terminal node and get the final reward from the environment. With AlphaGo, this entire step is skipped because instead of simulating the entire environment, you simply use a learned value function to "guess" the value of the newly expanded node. This is much, much faster, because you don't have to simulate the environment anymore, might actually provide better estimates of the value (especially if you have very complex environments, where random walks don't give you much information) and it allows you to use whatever information you learned about the environment.

So, let's go through each step.

### Traversal

In traversal, we first need to find a so-called _leaf node_. A leaf node is a node in a tree, which has no children. To do that, we apply the following simple algorithm:


![Traversal](/posts/monte-carlo-tree-search/MCTS_Traversal.drawio.svg)

We can very easily check if a node is a leaf node, but what is the `select_child_node` function? This function decides, how we select the next child node to traverse through. Once we reach a leaf node, we will use that to either expand the tree or perform a rollout. More on that later.

So, how do we choose this function? Well, it should be a function which takes in a list of nodes and returns the one it deems most appropriate, e.g.

- Upper Confidence Bound 1 (UCB1):
  Provides a good mix for exploration and exploitation
- Epsilon-Greedy:
  A simple approach where the best-known action is chosen most of the time, but with a small probability ε, a random action is selected.
- *Your own fancy shmancy action selection function*

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
