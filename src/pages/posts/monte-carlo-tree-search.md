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
You actually don't _have to_ perform a rollout. You do a rollout in the first place to estimate the _value of being in the current state_, to see what being in that state is worth.
But you could just as well use a neural network to estimate that value - in fact, this is what is typically used in MCTS these days.


### Our Environment

Before we actually start with implementing our MCTS algorithm, let's first have a look at our environment. The specific implementation doesn't matter, so I'll just leave it here for you
to toggle it into view if you want to.


<details>
<summary>Bandit Environment</summary>

```python

class BanditEnvironment:
    """
        This game tree looks like this:

            0
        / \\
        1   2
        / \\ / \\
        3   4 5  6
    """

    def __init__(self):
        self.tree = {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [], 4: [], 5: [], 6: []}
        self.current_state = np.array(0)

    def reset(self):
        self.current_state = np.array(0)
        return self.current_state

    def set_state(self, state):
        assert state in [0, 1, 2, 3, 4, 5, 6]
        self.current_state = state

    def step(self, action):
        if self.current_state in [3, 4, 5, 6]:
            return self.current_state, 0, True

        if action < 0 or action >= len(self.tree[int(self.current_state)]):
            raise ValueError("Invalid action")

        self.current_state = self.tree[int(self.current_state)][action]

        done = self.current_state in [3, 4, 5, 6]
        reward = 1 if self.current_state == 6 else 0

        return self.current_state, reward, done

    def render(self):
        print(f"Current state: {self.current_state}")

    @staticmethod
    def get_future_value(state):
        if state == 2:
            return 0.5
        elif state == 6:
            return 1
        else:
            return 0
```
</details>

In our environment, we have 7 states in total and the game tree would look like this:

![Game Tree](/posts/monte-carlo-tree-search/game_tree.drawio.svg)

You have at each state two available actions: going left or right. As you can see, only choosing the 'going right' action at state $2$ gives you a
reward of $1$, else you get a reward of $0$. The states $3$, $4$, $5$ and $6$ are terminal states, meaning the episode ends if you reach any
of those states.

From this, it's clear with route you should take, which is
$$
0 \rightarrow 2 \rightarrow 6
$$

This is the optimal path which we're trying to find using MCTS.

With that in mind, let's look at each of the steps from the MCTS algorithm and how we could implement them.

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
