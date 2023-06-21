---
layout: post
title:  "Generalizing Centipede Game States for Reinforcement Learning"
date:   2023-06-19 10:59:39 -0500
categories: machine-learning reinforcement-learning
---

## Overview

Reinforcement learning algorithms like Q-learning typically find an optimal policy for some Markov decision process by storing and updating a table of values used to map states to optimal actions[^3]. The most interesting MDPs require large or even infinite state-action spaces. It's intractable in these cases to enumerate and store the entire state-action space in a table. Moreover, many states in a large state-action space are similar and map to the same optimal action. One approach to handling these large spaces is to generalize the state-action space[^1]. A generalization of the state-action space condenses the amount of information needed to learn an optimal policy for the MDP.

{:refdef: style="text-align: center;"}
![state_cluster_examples](/assets/images/post-2023-06-19/state_cluster_examples.png){:style="text-align: center"}
{: refdef}

I use the Atari game Centipede to illustrate one approach to generalizing a large state space[^2][^4]. The observations in the RGB environment of Centipede are arrays with a shape of $$210 \text{ x } 160 \text{ x } 3$$ where each element takes on one of $$256$$ distinct values. This setup gives $$256^{210 \cdot 160 \cdot 3} \approx 3.877 \cdot 10^{242751}$$ possible distinct observations! We can reduce this massive number by switching to the RAM environment. Now each observation is a one dimensional array of size $$128$$ where each element has one of $$256$$ distinct values. Still, even using the RAM environment, there are $$256^{128} \approx 1.798 \text{ x } 10^{309}$$ possible observations. Constraints of time and space prohibit simply mapping each observation to an action. We need a more condensed representation of the state space. So instead we can develop a generalization of the observations from the RAM environment. Applying Q-learning to this generalization we find a policy that outperforms policies found using larger or random representations of the state space.

## Design

There are many approaches to generalizing the state-action space[^1]. We could generalize the entire state-action space. For example, deep Q-learning represents the Q-function with a neural network instead of a table. This representation enables learning an approximation of the entire state-action space using backpropagation. Alternatively, we can generalize the action space (agent's output) or the state space (agent's input). Here I choose to build a generalization over the state space.

We also need a sufficient set of state examples in order to learn a generalization of the state space. One way to produce this sample set is to uniformly sample from the entire state space. This method might prove problematic. Uniform sampling generates a sample set that considers each state equally. However, some states are more likely and more important than others. For example, in the Centipede game, states where the centipede is far away from the agent are relatively more likely than states where the centipede is very close. In order to account for these differences, we use empirical sampling. We run the MDP and allow the agent to explore the state space while recording the states it observes. The recorded states are used as the sample set.

Next, we choose a method to generalize the sample set of observed states. A generalization of the state space partitions the environment into buckets of states that share common characteristics. During Q-learning these buckets represent the same "prototypical" state. A successful generalization partitions these states in a way that suppress unnecessary details while emphasizing the most important. For example, the above observed states from the Centipede environment are grouped into the same prototypical state. Each of these states shared some important similar characteristics, e.g. the centipede is far away and broken into two pieces.

Choosing an appropriate method to find a partition of states depends on what assumptions are made about the environment and the sample set. For example, if the the environment is represented by a vector of boolean valued attributes, a model such as a decision tree might be appropriate for partitioning the states. In Centipede, the RAM environment is represented by a one dimensional vector of integers. Two vectors that are close in value represent similar game states. I chose k-means to find a specified number of average game states to use as prototypical states. The benefit of using k-means is its tendency to create well separated and balanced clusters. For example, clustering states into 20 prototypical states produced fairly balanced clusters. A plot of the first and second principal components also shows decent separation of the clusters.

{:refdef: style="text-align: center;"}
![state_cluster_comparison](/assets/images/post-2023-06-19/state_cluster_comparison.png){:style="text-align: center"}
{: refdef}

We also consider the level of granularity of the partition of states. If the partition is too granular, then the generalization will *exclude* important details of the environment. If the partition is too coarse, then the generalization fails to *include* important details about the environment. One method is to let the agent learn the appropriate level of granularity during training. This technique is known as adaptive resolution[^1]. Alternatively, if we know something about the environment and the right level of granularity, we can partition the state space prior to training. In the experiments here, I compare two levels of granularity. We partition the state space into 20 prototypical states and 200 prototypical states.

Generalizations of the state space sometimes need to combine supervised and unsupervised methods. In this case, we have a set of prototypical states produced by k-means clustering. However, the agent also needs a way to recognize which of the prototypical states it's in when it observes a new state. One simple solution is to use k-nearest neighbors model with $$k=1$$ to map new states to the nearest prototypical state. k-nearest neighbors is simple, interpretable, and doesn't require any training. So, the final design is summarized in the sequence diagram The `StateMap` component is the abstraction that represents the state generalization.

[![](https://mermaid.ink/img/pako:eNqNU9FuwiAU_RXCU006Y1trlUSftqdFX3xbmhiEq7K00AFd5oz_Pizaxdpk6xOce87h3lM4YaY4YIINfNQgGTwLute0zCVyHysESPu0WKwttbCkFUHbWhQ8MJf9xtCyKsCESG5YURsL2szj0cBrKbPi07HQTevh286Zvi6BSkPQTtigx6rr4-ke9Ounu86uOsRcy07uiRx69W0XzsEPSVoMqe07MIseDO4H6cmm0sAFs4HaGtBOIpT8dxgroBqMXYHYH7ZKN6lsegwvCckraR517TsuvtoB3cy_LUtfc60rq2x1FIwWqPm7KLgPdPCQR-9hfck2fn_Emctc4hCXoEsquLuQpwueY3uAEnJM3JLDjtaFzXEuz45Ka6vWR8kwsbqGENcVd4bX-4vJjhamRV-4sEq3YEXlm1LlTQlNdekfQvMeGgomJ_yFSZxmw-k0zeJpHCezOBmH-OjQaDpMk2yUROkkSqNJPD6H-LsxHQ0noyxKk7ErZbNZEk3OP-wfJM0?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNqNU9FuwiAU_RXCU006Y1trlUSftqdFX3xbmhiEq7K00AFd5oz_Pizaxdpk6xOce87h3lM4YaY4YIINfNQgGTwLute0zCVyHysESPu0WKwttbCkFUHbWhQ8MJf9xtCyKsCESG5YURsL2szj0cBrKbPi07HQTevh286Zvi6BSkPQTtigx6rr4-ke9Ounu86uOsRcy07uiRx69W0XzsEPSVoMqe07MIseDO4H6cmm0sAFs4HaGtBOIpT8dxgroBqMXYHYH7ZKN6lsegwvCckraR517TsuvtoB3cy_LUtfc60rq2x1FIwWqPm7KLgPdPCQR-9hfck2fn_Emctc4hCXoEsquLuQpwueY3uAEnJM3JLDjtaFzXEuz45Ka6vWR8kwsbqGENcVd4bX-4vJjhamRV-4sEq3YEXlm1LlTQlNdekfQvMeGgomJ_yFSZxmw-k0zeJpHCezOBmH-OjQaDpMk2yUROkkSqNJPD6H-LsxHQ0noyxKk7ErZbNZEk3OP-wfJM0)

## Results

TODO: summarize results

{:refdef: style="text-align: center;"}
![mean_scores_and_steps](/assets/images/post-2023-06-19/mean_scores_and_steps.png){:style="text-align: center"}
{: refdef}

{:refdef: style="text-align: center;"}
![max_min_scores_steps](/assets/images/post-2023-06-19/max_min_scores_steps.png){:style="text-align: center"}
{: refdef}

## Summary

The code for the experiments can be found on [github](https://github.com/jwplatta/centipede).

## References

[^1]: Kaelbing, L. P., Littman, M. L., & Moore, A. W. "Reinforcement Learning: A Survey." Journal of Artificial Intelligence Research, 4 (1996), 237-285.
[^2]: Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M. J., & Bowling, M. "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." Journal of Artificial Intelligence Research, 61, 523-562 (2018).
[^3]: Barto, Andrew G., and Richard S. Sutton. "Reinforcement Learning: An Introduction." MIT Press, 2018.
[^4]: [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)