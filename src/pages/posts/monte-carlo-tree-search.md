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
  UCB1(s) = V(s) + C \sqrt{\frac{ln N(s)}{n(s)}},
$$
where $V(s)$ is the average value of the state $s$, $C$ is a constant to balance exploration and exploitation, $N(s)$ is the number of visits to the **parent** node of $s$ and $n(s)$ is the number of visits to the state $s$.
