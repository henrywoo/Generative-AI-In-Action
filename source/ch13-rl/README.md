# Reinforcement Learning

![](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

from https://spinningup.openai.com/en/latest/index.html

https://www.zhihu.com/column/c_1215667894253830144

## What is Q and V?

https://zhuanlan.zhihu.com/p/109217883

![](https://pic1.zhimg.com/70/v2-44177b73b63d6ee182d574a1aeaa28ac_1440w.avis?source=172ae18b&biz_tag=Post)

## Proximal Policy Optimization(PPO)

https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8


## Others

### What is Inverse Reinforcement Learning (IRL) and Behavioral Cloning (BC)?

**Inverse Reinforcement Learning (IRL)**

* **Premise:**  Instead of directly being provided with a reward function, IRL figures out the reward function behind an expert's behavior. It's like watching someone play a game really well and trying to infer what goals they're trying to achieve.
* **How It Works**
    1. **Observes Expert:** IRL observes the expert's actions and the corresponding state transitions of the environment.
    2. **Guesses Reward Function:** It iteratively tries to find a reward function that when used by a reinforcement learning agent, would produce behavior similar to the expert's.
    3. **Learns the Policy:** Once a reasonable reward function is inferred, the IRL algorithm also learns a policy that tries to maximize that reward.

* **Use Cases:**
    * **Robotics:** Learning complex tasks from human demonstrations.
    * **Self-driving cars:** Inferring the implicit driving preferences of a human to train an autonomous car.
    * **Games:** Learning how to play games just by observing expert players.

**Behavioral Cloning (BC)**

* **Premise:** BC is a straightforward supervised learning approach. It directly mimics an expert's actions without trying to understand the underlying reward function. It's like a student copying the solutions from an answer key without fully understanding the problem-solving process.
* **How It Works**
    1. **Collects Demonstrations:** BC gathers a dataset of state-action pairs from the expert.
    2. **Trains a Model:**  A supervised learning model is trained to predict the expert's action for a given state.

* **Use Cases:**
    * **Simple Imitation:** Tasks where the logic behind the expert's decisions might be complex but direct observation of actions is sufficient.

**Key Differences & Considerations**

* **Generalization:** IRL attempts to learn the underlying goals, which could potentially lead to better generalization to unseen situations than BC.
* **Reward Shaping:** IRL provides an inferred reward function. This reward function can be used for further learning or shaping behaviors, something BC doesn't offer.
* **Complexity:** IRL is generally computationally more complex than BC due to the process of trying to reverse-engineer the reward function.


## What is policy gradient theorem? 

The policy gradient theorem provides the theoretical foundation for learning policies in reinforcement learning.

## What are behavior policy and target policy?

In reinforcement learning, behavior policy and target policy are two distinct concepts that play a crucial role, particularly in off-policy learning methods.

**Behavior Policy:**

* **The policy used to generate the data:** This is the policy that the agent actually follows while interacting with the environment. It determines which actions the agent takes in different states.
* **Focus on exploration:** The behavior policy is often designed to be exploratory, meaning it may take random or suboptimal actions to gather a diverse set of experiences. This helps the agent learn about the environment and potential rewards.
* **Not necessarily the optimal policy:** The behavior policy doesn't have to be the best possible policy. It's simply the policy that the agent uses to collect data.

**Target Policy:**

* **The policy being learned and improved:** This is the policy that the reinforcement learning algorithm is trying to optimize. It represents the agent's best understanding of how to act in different states to maximize rewards.
* **May be greedy or exploratory:** The target policy can be greedy (always choosing the action with the highest estimated value) or incorporate some degree of exploration to continue learning.
* **Updated based on experience:** The target policy is iteratively updated based on the experiences gathered by the behavior policy. This allows the agent to learn from its interactions with the environment and gradually improve its decision-making.

**Why Have Two Policies?**

The separation of behavior and target policies allows for greater flexibility and efficiency in reinforcement learning.

* **Learning from diverse experiences:** The behavior policy's exploration ensures that the agent gathers a wide range of experiences, even if those experiences aren't optimal. This diverse data helps the agent learn a more robust and generalizable target policy.
* **Learning from other agents:** In some cases, the behavior policy can be based on demonstrations from expert agents or other sources of data. This allows the agent to learn from the experience of others without having to directly follow their actions.
* **Efficient use of data:** By reusing data collected by the behavior policy, off-policy algorithms can learn more efficiently than on-policy algorithms, which can only learn from data generated by the current policy.

**Example: Q-Learning**

In Q-learning, a classic off-policy algorithm, the agent follows a behavior policy (often an epsilon-greedy policy) that balances exploration and exploitation. However, the Q-values are updated based on the maximum possible reward achievable from the next state, assuming the agent follows a greedy policy (the target policy). This allows the agent to learn the optimal policy even while exploring.



## Reference Links

- RLHF场景下的PPO算法的来龙去脉 https://www.zhihu.com/tardis/zm/art/631338315?source_id=1003
- 影响PPO算法性能的10个关键技巧（附PPO算法简洁Pytorch实现） https://zhuanlan.zhihu.com/p/512327050
- Spinning Up in Deep RL https://spinningup.openai.com/en/latest/
- 【强化学习2】Policy Gradient https://zhuanlan.zhihu.com/p/66205274
- 白话强化学习 https://www.zhihu.com/column/c_1215667894253830144
- DPO替代RLHF可造成多一倍的性能损失 https://zhuanlan.zhihu.com/p/673047773
- 从0开始实现LLM：7、RLHF/PPO/DPO原理和代码简读 https://zhuanlan.zhihu.com/p/686217468
- 强化学习中的重要性采样(Importance Sampling) https://zhuanlan.zhihu.com/p/371156865
- 重要性采样(Importance Sampling)详细学习笔记 https://zhuanlan.zhihu.com/p/342936969

