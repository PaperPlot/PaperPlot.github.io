---
title: "Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs"
date: 2024-12-12 17:40:14.112627
# weight: 1
# aliases: ["/first"]
tags: ['reinforcement learning', 'Markov decision processes', 'sample complexity', 'average reward']
author: "Me"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: ""
canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/<path_to_repo>/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

# Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs

## TLDR

This paper presents groundbreaking results on the sample complexity of learning near-optimal policies in average-reward Markov Decision Processes (MDPs) using a generative model. The key contributions are:

1. For weakly communicating MDPs, an optimal sample complexity bound of $\widetilde{O}(SA\frac{\mathsf{H}}{\varepsilon^2})$ is established, where $\mathsf{H}$ is the span of the optimal bias function.

2. For general (multichain) MDPs, a new transient time parameter $\mathsf{B}$ is introduced, leading to a sample complexity bound of $\widetilde{O}(SA\frac{\mathsf{B} + \mathsf{H}}{\varepsilon^2})$.

3. Improved bounds for $\gamma$-discounted MDPs are developed, showing $\widetilde{O}(SA\frac{\mathsf{H}}{(1-\gamma)^2\varepsilon^2})$ and $\widetilde{O}(SA\frac{\mathsf{B} + \mathsf{H}}{(1-\gamma)^2\varepsilon^2})$ sample complexities for weakly communicating and general MDPs respectively.

These results significantly advance our understanding of sample complexity in reinforcement learning, particularly for average-reward settings.

## Introduction

Reinforcement learning (RL) has seen remarkable successes in various sequential decision-making problems. As empirical achievements mount, there's an increasing need for theoretical understanding of RL algorithms and their fundamental limits. In this blog post, we'll dive deep into a recent paper by Matthew Zurek and Yudong Chen from the University of Wisconsin-Madison that tackles a foundational problem in RL: the sample complexity of learning near-optimal policies in average-reward Markov Decision Processes (MDPs) using a generative model.

The paper, titled "Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs," makes significant strides in understanding the sample complexity for both weakly communicating and general (multichain) MDPs. Let's unpack the key concepts, methodologies, and results presented in this work.

## Background: MDP Formulations and Reward Criteria

Before diving into the main results, it's crucial to understand the different MDP formulations and reward criteria considered in reinforcement learning:

1. **Finite Horizon Total Reward**: $\mathbb{E}^\pi\big[ \sum_{t=0}^T R_t\big]$
   This criterion measures the expected sum of rewards over a fixed number of time steps T.

2. **Infinite Horizon Discounted Reward**: $\mathbb{E}^\pi\left[ \sum_{t=0}^\infty \gamma^t R_t\right]$
   Here, future rewards are discounted by a factor $\gamma < 1$, emphasizing near-term rewards.

3. **Average Reward**: $\lim_{T \to \infty} (1/T) \mathbb{E}^\pi \bigl[ \sum_{t=0}^{T-1} R_t\bigr]$
   This criterion focuses on the long-term average performance of a policy.

The authors argue that the average reward criterion is often more appropriate for evaluating long-term performance, as it doesn't artificially discount future rewards or limit the horizon.

## The Sample Complexity Problem

The core problem addressed in this paper is determining the number of samples needed to learn a near-optimal policy in an MDP using a generative model. A generative model allows the algorithm to obtain independent samples of the next state given any initial state and action.

While sample complexity has been extensively studied for finite horizon and discounted reward settings, the average reward setting has remained an open problem. This paper aims to resolve this gap, particularly for weakly communicating MDPs and general (multichain) MDPs.

## Key Concepts and Parameters

To understand the results, we need to familiarize ourselves with some key parameters:

1. **Span of the optimal bias function** ($\mathsf{H}$): This is defined as $\mathsf{H} := \spannorm{h^\star}$, where $h^\star$ is the bias function of the optimal policy.

2. **Transient time parameter** ($\mathsf{B}$): A new parameter introduced for general MDPs, capturing the expected time spent in transient states.

3. **State-action space cardinality** ($SA$): The product of the number of states and actions in the MDP.

4. **Accuracy parameter** ($\varepsilon$): The desired accuracy of the learned policy compared to the optimal policy.

## Main Results

### 1. Weakly Communicating MDPs

For weakly communicating MDPs, the authors establish a sample complexity bound of:

$$\widetilde{O}\left(SA\frac{\mathsf{H}}{\varepsilon^2} \right)$$

This bound is minimax optimal (up to logarithmic factors) in all parameters $S$, $A$, $\mathsf{H}$, and $\varepsilon$. It improves upon existing work that either assumed uniformly bounded mixing times for all policies or had suboptimal dependence on the parameters.

### 2. General (Multichain) MDPs

For general MDPs, which may not be weakly communicating, the authors introduce the transient time parameter $\mathsf{B}$ and establish a sample complexity bound of:

$$\widetilde{O}\left(SA\frac{\mathsf{B} + \mathsf{H}}{\varepsilon^2} \right)$$

They also prove a matching (up to logarithmic factors) minimax lower bound, demonstrating the tightness of their result.

### 3. Improved Bounds for Discounted MDPs

As part of their analysis, the authors develop improved bounds for $\gamma$-discounted MDPs:

- For weakly communicating MDPs: $\widetilde{O}\left(SA\frac{\mathsf{H}}{(1-\gamma)^2\varepsilon^2} \right)$
- For general MDPs: $\widetilde{O}\left(SA\frac{\mathsf{B} + \mathsf{H}}{(1-\gamma)^2\varepsilon^2} \right)$

These bounds are significant because they circumvent the well-known minimax lower bound of $\widetilde{\Omega}\left(\frac{SA}{(1-\gamma)^3\varepsilon^2} \right)$ for $\gamma$-discounted MDPs, establishing a quadratic rather than cubic horizon dependence for a fixed MDP instance.

## Methodology and Key Insights

The authors' approach involves several novel techniques and insights:

1. **Reduction to Discounted MDPs**: The authors use a reduction from average-reward MDPs to discounted MDPs, but with crucial improvements over previous work.

2. **Improved Variance Analysis**: A key technical contribution is the development of tighter bounds on certain instance-dependent variance parameters. This is achieved through:
   - A new multistep variance Bellman equation
   - Recursive application to bound the variance of near-optimal policies

3. **Transient State Analysis**: For general MDPs, the authors develop new techniques to bound the total variance contribution from transient states, addressing challenges not present in the weakly communicating setting.

4. **Lower Bound Construction**: The authors provide a clever construction to prove the minimax lower bound for general MDPs, demonstrating the necessity of the $\mathsf{B}$ parameter.

Let's dive deeper into some of these methodological aspects.

### Multistep Variance Bellman Equation

A key technical tool developed in the paper is the multistep variance Bellman equation. For any integer $T \geq 1$ and any deterministic stationary policy $\pi$, we have:

$$\text{Var}^\pi\left[ \sum_{t=0}^{\infty} \gamma^t R_t  \right] = \text{Var}^\pi\left[ \sum_{t=0}^{T-1} \gamma^t R_t + \gamma^T V_\gamma^\pi(S_T) \right] + \gamma^{2T} P_\pi^T  \text{Var}^\pi\left[ \sum_{t=0}^{\infty} \gamma^t R_t  \right]$$

This equation allows for a more refined analysis of the variance terms, leading to tighter bounds.

### Variance Bounds for Optimal Policies

For the optimal policy $\pi^\star_\gamma$ in a weakly communicating discounted MDP, the authors show that:

$$\|\text{Var}^{\pi^\star_{\gamma}}\left[ \sum_{t=0}^{\infty}\gamma^t R_t \right]\|_\infty \leq 5 \frac{\mathsf{H}}{1-\gamma}$$

This bound is crucial for obtaining the improved sample complexity results.

### Transient State Analysis for General MDPs

For general MDPs, the authors introduce a decomposition of the transition matrix $P_\pi$ into recurrent and transient states:

$$P_\pi = \begin{bmatrix}
    X_\pi & 0 \\
    Y_\pi & Z_\pi
\end{bmatrix}$$

This decomposition allows for a more nuanced analysis of the variance contributions from transient states, leading to the introduction and utilization of the $\mathsf{B}$ parameter.

## Algorithmic Approach

The authors present two main algorithms:

1. **Algorithm for Discounted MDPs**: This algorithm (Algorithm 1 in the paper) uses a perturbed empirical model-based planning approach. It constructs an empirical transition kernel $\hat{P}$ using samples from the generative model and solves the resulting perturbed empirical MDP.

2. **Average-to-Discount Reduction**: Algorithm 2 reduces the average-reward MDP problem to a discounted MDP problem by setting an appropriate discount factor and then calling Algorithm 1.

The key to the algorithms' success is the careful choice of parameters, particularly the discount factor and perturbation level, which are set based on the span $\mathsf{H}$ and transient time $\mathsf{B}$ parameters.

## Implications and Future Directions

The results presented in this paper have several important implications:

1. **Optimal Bounds for Weakly Communicating MDPs**: The $\widetilde{O}(SA\frac{\mathsf{H}}{\varepsilon^2})$ bound for weakly communicating MDPs resolves a long-standing open problem, providing the first minimax optimal result in terms of all relevant parameters.

2. **New Understanding of General MDPs**: The introduction of the $\mathsf{B}$ parameter and the corresponding bounds for general MDPs open up new avenues for understanding and analyzing multichain MDPs.

3. **Improved Discounted MDP Analysis**: The quadratic horizon dependence for discounted MDPs (as opposed to the cubic dependence in previous work) provides new insights into the structure of these problems.

4. **Bridging Theory and Practice**: The span-based approach and the consideration of general MDPs bring theoretical analysis closer to practical RL scenarios, where the MDP structure may not always be well-behaved.

Future research directions suggested by this work include:

- Developing algorithms that can adapt to unknown $\mathsf{H}$ and $\mathsf{B}$ parameters
- Extending the analysis to model-free settings or settings with function approximation
- Investigating the implications of these results for online reinforcement learning algorithms

## Conclusion

The paper "Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs" represents a significant advance in our theoretical understanding of reinforcement learning. By providing tight sample complexity bounds for both weakly communicating and general MDPs in the average-reward setting, it fills a crucial gap in the literature.

The authors' novel techniques, particularly their refined variance analysis and treatment of transient states, offer new tools for analyzing RL algorithms. Moreover, their improved bounds for discounted MDPs challenge existing notions about the fundamental limits of these problems.

As we continue to push the boundaries of reinforcement learning in complex, real-world environments, theoretical results like these provide invaluable guidance. They help us understand the fundamental limits of what is achievable and point the way toward more efficient and effective algorithms.

For researchers and practitioners in the field of reinforcement learning, this paper is a must-read. It not only resolves long-standing open problems but also opens up new avenues for future research in both theory and applications of RL.

## References

1. Kearns, M., & Singh, S. (1998). Finite-Sample Convergence Rates for Q-Learning and Indirect Algorithms. Advances in Neural Information Processing Systems, 11.

2. Azar, M. G., Munos, R., & Kappen, H. J. (2013). Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine Learning, 91(3), 325-349.

3. Jin, Y., & Sidford, A. (2021). Towards Tight Bounds on the Sample Complexity of Average-reward MDPs. arXiv preprint arXiv:2106.07046.

4. Wang, J., Wang, M., & Yang, L. F. (2022). Near Sample-Optimal Reduction-based Policy Learning for Average Reward MDP. arXiv preprint arXiv:2212.00603.

5. Li, G., Wei, Y., Chi, Y., Gu, Y., & Chen, Y. (2020). Breaking the Sample Size Barrier in Model-Based Reinforcement Learning with a Generative Model. Advances in Neural Information Processing Systems, 33, 12861-12872.

