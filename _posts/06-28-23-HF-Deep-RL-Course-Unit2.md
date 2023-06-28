---
layout: post
comments: true
title:  "HuggingFace Deep RL Course Notes - Unit 2 (Work in Progress)"
excerpt: ""
date:   2023-06-28 10:00:00
mathjax: true
author: Zach Wimpee
thumbnail: /assets/hf.png
---

## Introduction
These post are going to be slightly different, as I am going to be using them as a sort of living document to record my notes from the [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course). I will be updating this post as I work through the course, so check back often for updates!

You can the notes for Unit 1 [here](https://zwimpee.github.io/2023/06/27/HF-Deep-RL-Course-Unit1/)

### Thoughts for Potential Projects and Further Areas of Interest
1.  Are these value functions differentiable? I am assuming so, but since we are dealing with discrete timesteps I am not sure what formulation is needed to take the derivative of these functions. I may be overthinking this, but as we more explicitly define the value functions throughout this unit I am going to keep this question in mind.

## Unit 2: Introduction to Q-Learning
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

For reference, I am including a glosarry of terms and concepts that are either introduced in this unit, or are important to restate here due to their application to the material we will be covering.

> ## Unit 2 - Glossary
>
> ### Two Types of Value Functions
>
> #### 2.1 State-Value Functions
>
> $$
> V^{\pi}(s) = \mathbb{E}_{\pi}[R_t | s_t = s]
> $$
>
> #### 2.2 Action-Value Functions
>
> $$
> Q^{\pi}(s, a) = \mathbb{E}_{\pi}[R_t | s_t = s, a_t = a]
> $$
>
> ### 2.3 Optimal Policy
>
> $$
> \pi^*(s) = \text{argmax}_{a} Q^*(s, a) \quad \text{for all}\ s
> $$
>
> ### 2.4 Bellman Equation
>
> $$
> V^{\pi}(s) = \mathbb{E}_{\pi}[R_t | s_t = s] = \mathbb{E}_{\pi}[r_{t+1} + \gamma V^{\pi}(s_{t+1}) | s_t = s]
> $$
>
>
> ### 2.5 Q-Learning - Update Rule
>
> $$
> Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \text{max}_{a'} Q(s', a') - Q(s, a)]
> $$
>
> ### 2.6 Epsilon Greedy Policy 
>
> $$
> \pi(a|s) = 
> \begin{cases} 
> 1 - \epsilon + \frac{\epsilon}{|A|}, & \text{if}\ a = \text{argmax}_{a'} Q^{\pi}(s, a') \\
> \frac{\epsilon}{|A|}, & \text{otherwise}
> \end{cases}
>
> $$
>

## Unit 2 - Content
### Value-Based Methods

<div class="imgcap_noborder">
<img src="/assets/vbm-1.jpg" width="80%">
</div>

In value-based methods, we learn a value function that maps a state to the expected value of being at that state. This value function is typically denoted as $$V(s)$$.

Borrowing directly from HuggingFace:
> *The value of a state is the expected discounted return the agent can get if it starts at that state and then acts according to our policy.*

That is, given some state $$S_t$$, the value function $$V(s)$$ will return the expected value of being at that state.

Our goal remains the same; we want to find the optimal policy $$\pi^*$$ that maximizes the expected return $$G_t$$. However, we are not doing so directly, but are instead training a model on a value function for a given state, and define the policy in terms of the value function.

From the course material:

> "In fact, most of the time, in value-based methods, you’ll use an Epsilon-Greedy Policy that handles the exploration/exploitation trade-off; we’ll talk about this when we talk about Q-Learning in the second part of this unit."

What this "Epsilon-Greedy Policy" does, without going into too much detail here, is that it allows us to explore the environment, while still exploiting our immediate surroundings in order to maximize the expected return.

### Two types of value-based methods

<div class="imgcap_noborder">
<img src="/assets/two-types.jpg" width="80%">
</div>

In value-based methods for finding the optimal policy, we have two types of value functions:
  - **State-Value Functions**
    - For the state-value function, we calculate the value of each state $$S_t$$,

<div class="imgcap_noborder">
<img src="/assets/state-value-function-1.jpg" width="80%">
</div>

<div class="imgcap_noborder">
<img src="/assets/state-value-function-2.jpg" width="80%">
</div>


#### **Action-Value Functions** 
For the action-value functions, we assign a value to each tuple $$(S_t, A_t)$$, where $$A_t$$ is the action taken between possible states.


<div class="imgcap_noborder">
<img src="/assets/action-state-value-function-1.jpg" width="80%">
</div>

<div class="imgcap_noborder">
<img src="/assets/action-state-value-function-2.jpg" width="80%">
</div>
    

The state-value functions contain less information than the action-value functions, but finding the action-value function explicitly would be much more computationally intensive. Therefore, we are going to introduce our first named equation in the course, the ***Bellman Equation***.


### Bellman Equation
We will go ahead and just state the equation, and then we can begin breaking it down and dissecting how exactly it works.
For now, just know that it is a recursive function that approximates the computationally expensive state-action-value functions:

$$
V^{\pi}(s) = \mathbb{E}_{\pi}[R_t | s_t = s] = \mathbb{E}_{\pi}[r_{t+1} + \gamma V^{\pi}(s_{t+1}) | s_t = s]
$$

<div class="imgcap_noborder">
<img src="/assets/bellman4.jpg" width="80%">
</div>

### Two Learning Strategies

#### Monte-Carlo Learning

This type of learning involves looking back after an episode and adjusting the value function based on the actual return. 

For a state-value function, we can express this in the following equations:

$$
V^{\pi}(s) \leftarrow V^{\pi}(s) + \alpha[G_t - V^{\pi}(s)]
$$

Where here,

- $$G_t$$ is the discounted return at time step $$t$$
- $$V^{\pi}(s)$$ is the value function, which maps a given state to a certain value.

Then intuitively we can interpret this equation as follows:

- We first play out 1 full episode, and compute the discounted cumulative return, $$G_t$$.
- At each timestep $$t$$, we adjust the value function by some coefficients $$\alpha$$ and the difference between the total return $$G_t$$ over the entire episode minus the value function at that state $$V^{\pi}(s)$$.

We can see then why we have to wait an entire episode before adjusting the value function for each state $$G_t$$. We need to know the total return for the entire episode before we can adjust the value function for each state. The amount by which each state is adjusted by the amount it *didn't* contribute to $$G_t$$, scaled by a coefficient $$\alpha$$. So that highest values for each of the states will be adjusted less than the ones that contributed the least to $$G_t$$. The factor $$\alpha$$ is the learning rate, and adds an additional parametrizable constraint we can adjust in order to achieve our ultimate goal of finding the optimal policy $$\pi^*$$.

Extending this to the state-action-value function $$Q^{\pi}(s, a)$$, we can express the Monte-Carlo learning equation as follows:

$$
Q^{\pi}(s, a) \leftarrow Q^{\pi}(s, a) + \alpha[G_t - Q^{\pi}(s, a)]
$$

Where here the parameters of this equation follow naturally from the parameters we defined for the state-value function $$V^{\pi}(s)$$.

Going back to a state-value function $$V^{\pi}(S_t)$$, we will borrow some nice summarization and illustration from the Hugging Face course:


> # Example of Monte-Carlo Learning
>  <div class="imgcap_noborder">
>  <img src="/assets/MC-2.jpg">
>  </div>
> - We will always start each episode at the same starting point
> - The agent takes actions using the policy. For instance, using an Epsilon Greedy Strategy, a policy that alternates between exploration (random actions) and exploitation.
> - We get the reward and the next state.
> - We terminate the episode if the cat eats the mouse or if the mouse moves > 10 steps.
> - At the end of the episode, we have a list of State, Actions, Rewards, and Next States tuples For instance [[State tile 3 bottom, Go Left, +1, State tile 2 bottom], [State tile 2 bottom, Go Left, +0, State tile 1 bottom]…]
> - The agent will sum the total rewards $$G_t$$ (to see how well it did).
> - It will then update $$V(S_t)$$ based on the formula
>  <div class="imgcap_noborder">
>  <img src="/assets/MC-3.jpg">
>  </div>
> - The agent will then start a new episode and repeat the process.
> - The process can be summarized by the following illustration:
>  <div class="imgcap_noborder">
>  <img src="/assets/MC-3p.jpg">
>  </div>

Now, we will move onto a more scalable solution, Temporal Difference Learning.

#### Temporal Difference Learning

# Temporal Difference Learning

*will be getting the rest of this post written shortly, stay tuned...*



#### References
- [1][HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course)
- [2][Wikipedia: Bellman Equation](https://en.wikipedia.org/wiki/Bellman_equation)
- [3][Towards Data Science Post: Monte-Carlo Learning](https://towardsdatascience.com/monte-carlo-learning-b83f75233f92)