---
layout: post
comments: true
title:  "HuggingFace Deep RL Course Notes - Unit 2"
excerpt: ""
date:   2023-06-27 10:00:00
mathjax: true
author: Zach Wimpee
thumbnail: /assets/intro/thumbnail.png
---

# HuggingFace Deep RL Course Notes
These post are going to be slightly different, as I am going to be using them as a sort of living document to record my notes from the [HuggingFace Deep RL Course](https://huggingface.co/course/chapter1). I will be updating this post as I work through the course, so check back often for updates!



# Unit 2: Introduction to Q-Learning
In this unit, we will explore Q-Learning, a popular value-based method for solving RL problems.

This will involve us implementing an RL-agent from scratch, in 2 environments:
- Frozen-Lake-v1 (non-slippery version): where our agent will need to go from the starting state (S) to the goal state (G) by walking only on frozen tiles (F) and avoiding holes (H).
- An autonomous taxi: where our agent will need to learn to navigate a city to transport its passengers from point A to point B.

Pulling directly from the course material:
> Concretely, we will:
    > - Learn about value-based methods.
    > - Learn about the differences between Monte Carlo and Temporal Difference Learning.
    > - Study and implement our first RL algorithm: Q-Learning.
> This unit is fundamental if you want to be able to work on Deep Q-Learning: the first Deep RL algorithm that played Atari games and beat the human level on some of them (breakout, space invaders, etc).
>
> So let’s get started! 🚀

## Value-Based Methods
![Alt text](../assets/vbm-1.jpg)

In value-based methods, we learn a value function that maps a state to the expected value of being at that state. This value function is typically denoted as $V(s)$.

Borrowing directly from HuggingFace:
>*The value of a state is the expected discounted return the agent can get if it starts at that state and then acts according to our policy.*

That is, given some state $s_t$, the value function $V(s)$ will return the expected value of being at that state.



### Two types of value-based methods
There are two types of value-based methods:
- Monte Carlo methods
- Temporal Difference methods


### Monte Carlo Methods
Monte Carlo methods are a type of value-based method that learn directly from episodes of experience. That is, Monte Carlo methods learn from complete episodes of experience, rather than step-by-step. We can more formally define Monte Carlo methods as follows:





#### References
- [HuggingFace Deep RL Course](https://huggingface.co/course/chapter1)