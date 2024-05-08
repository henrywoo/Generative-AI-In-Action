# Large Language Model

- Cast: LLM, Transformer
- Song: [Large Language Model](https://www.youtube.com/watch?v=59UIVmFkxbs), [Attention Is All You Need](https://www.youtube.com/watch?v=g_tvY_pVwKI)



LLMs are advanced artificial intelligence models trained on massive amounts of text data. They can generate realistic text, translate languages, write different kinds of creative content, and answer your questions in informative ways. An LLM's power comes from its training process.  It "learns" to understand and produce language by analyzing enormous amounts of text – think millions of books, articles, and code repositories. Most LLMs are built on a neural network architecture called a transformer. Transformers excel at handling sequential data (like words in a sentence) and pinpointing complex relationships within language.

LLMs streamline how we use search engines, get help from virtual assistants, and even create content. LLMs are still evolving, but the potential is huge. They could revolutionize education, healthcare, customer service, and more.

LLMs sometimes generate incorrect information or output text that seems logical but is factually wrong. It's essential to be aware of their limitations. The quality of data used for training greatly influences the LLM's abilities and potential biases.

## Model Basics


### Why Transformer is used for LLM, instead of CNN, RNN or MLP?

Here's a breakdown of why transformers are favored over CNNs, RNNs, and MLPs for building LLMs:

**Transformers and Their Advantages for LLMs**

* **Sequential Data Mastery:** Transformers excel at handling sequences, which is crucial for LLMs that deal with text. They can analyze the relationships between words across long distances within a sentence, unlike CNNs or MLPs that struggle with long-range dependencies.

* **Attention Mechanism:** A core strength of transformers is their attention mechanism. This allows the model to focus on specific parts of the input sequence that are most relevant to the current processing step. This is particularly beneficial for LLMs where understanding context is critical.

* **Parallel Processing Power:** Transformers are well-suited for parallel processing, which means they can efficiently train on massive datasets using multiple graphics processing units (GPUs) at once. This is essential for LLMs that require enormous amounts of training data.

**Why Other Architectures Fall Short for LLMs**

* **CNNs (Convolutional Neural Networks):** While CNNs are powerful for image recognition, they struggle with sequential data like text. Their strength lies in capturing local patterns, not long-range dependencies in sentences.

* **RNNs (Recurrent Neural Networks):** RNNs can handle sequences, but they suffer from the vanishing/exploding gradient problem, making it difficult to learn long-range dependencies. LSTMs and GRUs partially address this, but transformers offer a more efficient solution.

* **MLPs (Multi-Layer Perceptrons):** MLPs are simpler models that lack the capability to effectively capture the complex relationships between words in a sentence. They are not well-suited for the intricacies of natural language.

**In essence, transformers provide the ideal combination of:**

* **Sequential data processing**
* **Attention mechanism for context**
* **Parallel processing for efficient training**

These factors make them the preferred architecture for building powerful LLMs.

**Additional Points to Consider**

* **Research into Alternative Architectures:** While transformers are dominant, research is ongoing for alternative LLM architectures that address potential shortcomings of transformers, such as computational cost for very long sequences.

----

###  🌡️ What is temperatures and how does it work under the hood?

Temperature is a parameter used in language models, particularly in models that generate text, to control the **randomness of predictions** by scaling the logits (the outputs of the final linear layer) before applying the softmax function to convert these logits into probabilities. Here's a step-by-step explanation of how temperature affects the generation probability under the hood:

1. **Logits Calculation**: The model computes logits, which are essentially raw predictions of the next token given the input. These logits represent the unnormalized log probabilities of each token in the model's vocabulary.

2. **Scaling by Temperature**: The temperature parameter \( T \) is used to scale the logits. The formula used is:
   \[
   \text{adjusted logits} = \frac{\text{original logits}}{T}
   \]
   Here, \( T > 0 \). A temperature of 1 means no scaling; the logits are used as they are. Temperatures greater than 1 make the logits smaller, and temperatures less than 1 make the logits larger.

3. **Softmax Function**: After scaling the logits with the temperature, a softmax function is applied to convert these logits into probabilities. The softmax function is defined as:
   \[
   P_i = \frac{e^{\text{adjusted logits}_i}}{\sum_j e^{\text{adjusted logits}_j}}
   \]
   where \( P_i \) is the probability of the \( i \)-th token.

4. **Effect of Temperature**:
   - **High Temperature (\( T > 1 \))**: This makes the distribution of logits more uniform (closer to each other), leading to a flatter distribution of probabilities. As a result, the model generates text with more randomness and creativity, as less likely tokens gain a relatively higher probability of being chosen.
   - **Low Temperature (\( T < 1 \))**: This makes the distribution of logits more peaky as differences between logits are amplified. Consequently, the probability distribution becomes sharper, meaning the model's output becomes more deterministic and repetitive, favoring more likely tokens.
   - **Temperature of 1**: This retains the original logits, leading to a balance between randomness and determinism based on the model's original training.

Temperature effectively adjusts the "confidence" of the model's predictions, allowing for control over the diversity and creativity of the generated text. This makes it a crucial tool for fine-tuning the output of language models, especially in applications where varying levels of creativity or adherence to typical responses are desired.

----

### What is Chinchilla scaling Law? Why is it important?

In general, a neural model can be characterized by 4 parameters: size of the model, size of the training dataset, cost of training, performance after training. Each of these four variables can be precisely defined into a real number, and they are empirically found to be related by simple statistical laws, called "scaling laws".

In simpler terms, the Chinchilla scaling law for training Transformer language models suggests that when given an increased budget (in FLOPs), to achieve compute-optimal, the number of model parameters (N) and the number of tokens for training the model (D) should scale in approximately equal proportions. This conclusion differs from the previous scaling law for neural language models, which states that N should be scaled faster than D. The discrepancy arises from setting different cycle lengths for cosine learning rate schedulers. In estimating the Chinchilla scaling, the authors set the cycle length to be the same as the training steps, as experimental results indicate that larger cycles overestimate the loss of the models.

The Chinchilla scaling law is described in the paper titled "Training Compute-Optimal Large Language Models" by researchers at DeepMind. This paper presents a detailed analysis of how the scaling of data and model size affects the performance of large language models, leading to the development of the Chinchilla model. It provides insights into the optimal allocation of computing resources for training these models, emphasizing the importance of using larger datasets relative to model size for enhanced performance. This work has contributed significantly to the ongoing discussions and strategies around the development of AI language models.

**LLaMA3** (https://ai.meta.com/blog/meta-llama-3/):

> To effectively leverage our pretraining data in Llama 3 models, we put substantial effort into scaling up pretraining. Specifically, we have developed a series of detailed scaling laws for downstream benchmark evaluations. These scaling laws enable us to select an optimal data mix and to make informed decisions on how to best use our training compute. Importantly, scaling laws allow us to predict the performance of our largest models on key tasks (for example, code generation as evaluated on the HumanEval benchmark—see above) before we actually train the models. This helps us ensure strong performance of our final models across a variety of use cases and capabilities.
> 
> We made several new observations on scaling behavior during the development of Llama 3. For example, while the Chinchilla-optimal amount of training compute for an 8B parameter model corresponds to ~200B tokens, we found that model performance continues to improve even after the model is trained on two orders of magnitude more data. Both our 8B and 70B parameter models continued to improve log-linearly after we trained them on up to 15T tokens. Larger models can match the performance of these smaller models with less training compute, but smaller models are generally preferred because they are much more efficient during inference.

**QA**

* **Q:** With a 2T tokens for training, what is the optimal size of an LLM considering cost-cost efficiency?
* **A:** According to `Chinchilla Scaling Law`, the optimal model size is 2T/20 = 100B.


https://en.wikipedia.org/wiki/Neural_scaling_law
https://www.aisafetybook.com/textbook/2-4 
https://www.youtube.com/watch?v=joZaCw5PxYs&ab_channel=AICoffeeBreakwithLetitia
https://www.zhihu.com/question/628395521/answer/3270617687

----

### Why most LLM are decoder-only?

Here's a breakdown of the reasons why most large language models (LLMs) favor a decoder-only architecture:

**1. Task Suitability**

* **Causal Language Modeling:** The primary objective of LLMs has traditionally been generative text tasks. This means predicting the next word or token given a sequence of previous words. Decoder-only architectures are a natural fit for this causal language modeling setup, as they only have access to past context.
* **Efficiency:**  In tasks like translation or summarization, where the input and output sequences have strong dependencies, bidirectional attention (encoder-decoder) might be more beneficial.  However, pure generative text tasks benefit from the computational efficiency of decoder-only models.

**2. Training Advantages**

* **Parallelism:** Decoder-only models enable highly efficient parallelization during training. Since each position attends only to the past, computations for different tokens can happen simultaneously, leading to faster training times.
* **Data Availability:**  Massive text datasets are readily available. Training solely on this type of data aligns perfectly with the causal prediction capabilities of decoder-only models.

**3. The Low-Rank Issue**

![](https://d3i71xaburhd42.cloudfront.net/995afe47244913ac8d1b4f09bbfacd407f1b4a7b/4-Figure2-1.png)

* **Expressivity Concerns:** Bidirectional attention can introduce the low-rank problem, potentially reducing the LLM's ability to represent complex relationships in the input. Decoder-only LLMs avoid this issue, ensuring strong baseline performance.

**4. Performance Success**

* **Empirical Evidence:**  Decoder-only models like GPT-3 have achieved impressive results on various language tasks, demonstrating that they can learn rich linguistic representations even with the unidirectional constraint.
* **Refinement over Replacement:** Much of the recent research has focused on refining decoder-only LLMs (scaling, efficient attention mechanisms, etc.) rather than fundamentally shifting towards bidirectional architectures for pure language generation tasks.

**Important Considerations**

* **Not Universal:**  While decoder-only models dominate, there are scenarios where bidirectional attention (encoder-decoder) is beneficial.  Tasks that require understanding the entirety of the input sequence, like machine translation or question answering, often use encoder-decoder architectures.
* **Evolving Landscape:** Research is ongoing. New techniques for mitigating the limitations of bidirectional attention or hybrid approaches combining the strengths of both architectures could emerge in the future.

**In Summary**

The dominance of decoder-only LLMs stems from a combination of factors: their natural alignment with generative text tasks, training efficiency, avoidance of potential expressiveness limitations, and the sheer success they've achieved.

https://www.zhihu.com/question/588325646/answers/updated


### What is Group Query Attention?

![](https://picx.zhimg.com/70/v2-6f6e56cc3f801fa47831a295a0ced703_1440w.avis?source=172ae18b&biz_tag=Post)

- https://zhuanlan.zhihu.com/p/647130255 👍
- https://zhuanlan.zhihu.com/p/667259791


### What is RMSProp?

https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a

![](https://miro.medium.com/v2/resize:fit:640/format:webp/0*o9jCrrX4umP7cTBA)

### What is Prefix Decoder Architecture?

![](https://cdn.labellerr.com/language%20models-4/Screenshot%202023-05-21%20233029.webp)

https://www.labellerr.com/blog/exploring-architectures-and-configurations-for-large-language-models-llms/ 👍

### What is an MoE layer?

https://stackoverflow.blog/2024/04/04/how-do-mixture-of-experts-layers-affect-transformer-models/

https://github.com/XueFuzhao/OpenMoE


## LLM Finetuning

### What is ORPO?

During SFT, the probability of generating undesirable responses along with preferred ones also increases.

Preference alignment is then employed to address this issue. It aims to increase the likelihood of generating preferred responses and decrease the likelihood of generating rejected responses. Traditionally, preference alignment is achieved through techniques like Reinforcement Learning with Human Feedback (RLHF) or Direct Preference Optimization (DPO). However, these methods require a separate reference model, increasing computational complexity.

ORPO elegantly solves this problem by combining SFT and preference alignment into a single objective function. It modifies the standard language modeling loss by incorporating an odds ratio (OR) term. 



## LLM in Production

### Why BitLinear can quantize LLM model to 1.58 bit without much loss on performance?

Here's a breakdown of why BitLinear can quantize LLM models to 1.58 bits, along with the principles behind this technique:

**Understanding BitLinear**

* **Beyond 8-bit Quantization:** Traditional quantization often scales parameters to 8-bit integers for efficiency. BitLinear goes further by considering the importance of different parameters within the LLM. 
* **Ternary Representation:**  It assigns higher precision to parameters that have a larger impact on the model's output, and lower precision to those with less impact.  In BitLinear, weights are stored in a ternary format of [-1, 0, +1].
* **1.58 Bits on Average:**  Due to the mix of precisions in ternary representation, the *average* storage per parameter comes out to about 1.58 bits. 

**Why It Works**

1. **LLM Redundancy:** Large language models have an inherent level of redundancy.  Not every parameter is equally crucial for accurate output.
2. **Sparsity:**  Introducing 0 values in the ternary representation creates sparsity, further aiding computational efficiency.
3. **Simple Operations:**  Since matrix weights are limited to -1, 0, and +1, the underlying multiplications are replaced by simple additions and subtractions – these are much faster to compute.

**Results**

* **Significant Compression:** BitLinear can drastically reduce model size compared to standard 32-bit floating-point representations.
* **Memory Savings:**  This leads to reduced memory consumption, potentially allowing for larger models to be deployed on resource-constrained devices.
* **Computational Speedups:** The simplified operations in BitLinear can lead to faster inference times.

**Important Considerations**

* **Accuracy Tradeoff:**  While BitLinear achieves good compression, there is often a slight decrease in model accuracy compared to the full-precision version. Researchers continually fine-tune the method to minimize this gap.
* **Not a Universal Solution:**  The optimal quantization strategy depends on the specific model architecture and the task at hand.

**Paper Reading**

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)



## Prompt Engineering

Prompt engineering, then, is the practice of crafting a prompt whose completion contains a high quality answer to **whatever** your current question is, or a solution to your current problem.


### In-Context Learning (ICL)

**Definition**

* **Learning from Examples within the Prompt:** In-context learning is a technique where you provide a large language model (LLM) with a few examples of the task you want it to perform, directly within the input prompt itself. The model then learns to perform that  task *without* the need for extensive fine-tuning on a specific dataset. 
* **Adapting on the Fly:** ICL allows the model to adapt to new tasks or situations based on the examples you provide, giving it more flexibility than traditional supervised learning methods.

**How In-Context Learning Works**

1. **Task Description & Examples:**  You provide the LLM with a prompt that includes:
    *  A clear description of the task you want it to do.
    *  A few examples of input-output pairs that demonstrate how the task should be performed.
2. **Pattern Recognition:** The LLM examines the examples in your prompt, searching for patterns and relationships between the inputs and the desired outputs.
3. **Inference:** When you give the LLM a new input, it uses the patterns learned from the examples to infer the correct output, essentially performing the new task.

**Why In-Context Learning is Significant**

* **Zero-shot or Few-shot Learning:**  ICL allows LLMs to perform tasks they haven't been explicitly trained on (zero-shot), or to learn from far fewer examples than traditional training requires (few-shot).
* **No Fine-Tuning:**  You don't have to modify the model's massive set of parameters, saving time and computational resources.
* **Unlocking New Applications:**  ICL makes LLMs adaptable to a wider range of tasks without large-scale retraining, potentially opening up entirely new use cases. 

**Example: Sorting Lists**

**Input to the Model:**
```
1. Given the following lists, sort each one in ascending order:
   - List 1: [5, 2, 9, 1, 5]
   - List 2: [8, 12, 3, 0, 7]
   - List 3: [22, 13, 9, 5, 31]

2. Here are the sorted lists:
   - List 1: [1, 2, 5, 5, 9]
   - List 2: [0, 3, 7, 8, 12]
   - List 3: [5, 9, 13, 22, 31]
```

**Follow-up Input:**
```
Now sort this list: [4, 11, 8, 6, 3]
```

**Expected Output (by the model, using in-context learning):**
```
[3, 4, 6, 8, 11]
```

In this example, the model infers from the provided context that its task is to sort lists of numbers in ascending order. When given a new list, it applies the learned pattern from the earlier examples to produce the correct output, demonstrating in-context learning.

When language models like me process tasks such as sorting a list, we don't actually execute an algorithm like a traditional computer program would. Instead, we generate responses based on patterns and examples we've seen during our training. For instance, if a model is frequently exposed to tasks where lists are sorted, it learns to recognize this as a sorting task and mimics the pattern of sorting in its responses.

In the example of sorting lists, the model doesn’t technically use a specific sorting algorithm such as QuickSort or MergeSort. Rather, it predicts the most likely output (the sorted list) based on the training data it has seen that involves similar tasks. This process involves understanding the context of the question, recognizing it as a sorting task, and then generating a list that appears to be sorted in ascending order.

This is different from how a computer program sorts a list, where it would explicitly execute steps defined by a sorting algorithm to rearrange the items in the list into the correct order.

**Time Complexity & Comparison With Well-defined Sorting Algorithm**

The time complexity of sorting a list using a language model like GPT isn't straightforward to define in traditional computational terms such as those used for algorithms (like O(n log n) for MergeSort). This is because language models don't sort through computational steps or algorithms in the conventional sense.

When a language model processes a sorting task, it doesn't manipulate or iterate through the elements of the list as a sorting algorithm would. Instead, it generates an output based on the learned patterns from the training data. The model essentially "guesses" the sorted order based on its training on similar tasks. Thus, the efficiency of a language model in producing a sorted list isn't measured in terms of operations on elements of the list (as in traditional time complexity), but rather by how well it has been trained to recognize and produce patterns.

However, if we were to discuss the computational cost of generating a response by a language model, it would be related to the number of tokens processed and the operations involved in generating each token. This involves matrix multiplications and activations across the layers of the neural network. The actual computational complexity for generating each token can be considered in terms of the number of operations required per token, which depends on the model's architecture (e.g., number of layers, size of each layer). But this doesn't translate directly into the traditional time complexity metrics used for algorithms like sorting.

**Paper**:
- https://arxiv.org/pdf/2301.00234
