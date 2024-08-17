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

We're going to implement two versions of MCTS: one more object-oriented-_ish_ and one using arrays. The reason for that is that the OO version is easier to understand and it's easy to translate that to the efficient array version.

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
import numpy as np


def ucb1(avg_node_value, visits_parent, visits_node, exploration_exploitation_factor=2):
    return avg_node_value + exploration_exploitation_factor * np.sqrt(
        np.log(visits_parent) / visits_node
    )
```

Simple stuff. Now, how do we go about traversing the tree? Where is the tree anyway? What does the data structure implementation look like?

### A Node

Let's start with a node class and put in all the properties we already know a node should have.


```python
from beartype.typing import Any

class Node:
    index: int

    child_nodes: dict[int, "Node"]
    parent_node: "Node | None"

    visits: int
    value: float

    embedding: Any
```

A node will get more properties later, but these are the core ones we care about. The child nodes dictionary maps an action (we're dealing with discrete actions) of type integer to another node. We use the double quotes as a "forward reference" because the `Node` class hasn't been fully defined at that point.

Each `Node` will also have a reference to its parent but not all nodes will have a parent. Actually, only the root node will not have a parent and by checking if `node.parent is None` we can infer if the current node is the root node.

We also keep track of the number of visits as well as the node's average value. Lastly, each node will be "assigned a state" from our environment and that will be stored in the `embedding` field. In our example, that will be simply the index of the state (e.g. index $6$, which is the terminal state and gives a reward of 1).

Let's add an `__init__` and a `__repr__` and a helper function:

```python
class Node:
    index: int

    child_nodes: dict[int, "Node"]
    parent_node: "Node | None"

    visits: int
    value: float

    embedding: Any

    def __init__(self, parent: "Node | None", index: int, embedding: Any) -> None:
        self.parent = parent
        self.index = index
        self.embedding = embedding

        self.child_nodes = dict()
        self.visits, self.value = 0, 0

    def is_child_visited(self, action) -> bool:
        return action in self.child_nodes

    def __repr__(self) -> str:
        return f"[Index: {self.index}, Parent: {self.parent.index if self.parent is not None else None}, Value: {np.round(self.value, 2)}, Visits: {self.visits}]"
```

And now, we can initialise our root node:

```python
def get_root_node(env: BanditEnvironment) -> Node:
    obs = env.reset()
    return Node(parent=None, index=0, embedding=obs)


env = BanditEnvironment()
root_node = get_root_node(env)
```

### Tree Traversal

And with that, we can implement our tree traversal function. I'll paste it here first and then we will go over it step by step.

```python
def traversal(
    root_node: Node, max_depth: int, action_selection_fn: Callable[[Node, int], int]
) -> tuple[Node, int]:
    class TraversalState(NamedTuple):
        node: Node # the parent node
        next_node: Node | None # the child node
        action: int # the action to perform from the parent node
        depth: int # the current depth
        proceed: bool # whether or not to continue

    def _traversal(state: TraversalState) -> TraversalState:
        node = state.next_node # the current node is last iteration's next node
        action = action_selection_fn(node, state.depth)
        child_visited = node.is_child_visited(action)
        if not child_visited:
            next_node = None
        else:
            next_node = node.child_nodes[action]
        proceed = not child_visited and state.depth + 1 < max_depth

        return TraversalState(
            node=node,
            next_node=next_node,
            action=action,
            depth=state.depth + 1,
            proceed=proceed,
        )

    state = TraversalState(
        node=root_node, next_node=root_node, action=0, depth=0, proceed=True
    )

    while state.proceed:
        state = _traversal(state)

    return state.node, state.action
```

The `TraversalState` is simply a struct to keep track of the traversal. We initialize the first state like so:

```python
state = TraversalState(
    node=root_node, next_node=root_node, action=0, depth=0, proceed=True
)
```
For the first iteration, we don't care about `state.node` and if you're happy to ignore pyright, you might also set it to `None`, but we deeply care about pyright, so we won't (remember, that the `state.node` refers to the parent and that the root node has no parent). We set the `next_node` to the root node and in our traversal loop, we set the current node to be last iteration's `next_node`. That's why we have to assert that `node` is not `None`.

This means in the first iteration, `node = state.next_node` refers to what?

1) `None`
2) root
3) I don't know :(

The answer is: the root node! Now that we have the current node (which is the root in the first iteration), we use the `action_selection_fn` callable to select the next action. In our case, that is going to be the `UCB1` function. Once we have the action, we check whether or not the next state was visited or not. If it's not visited, we can stop right there, otherwise we select that action, increment the depth by 1 and then proceed with the next step. The while loop will end, once `proceed` is `False`. Finally, we will return the `node` (which is the parent) and the action we want to expand.
