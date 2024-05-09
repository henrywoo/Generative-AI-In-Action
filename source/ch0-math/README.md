# AI Basics

## Normalization

The term "norm" in the contexts of data scaling (such as normalization using min-max scaling or standardization) and in Layer Normalization (a technique used in deep learning) refers to different concepts and processes. Here’s a brief overview of how "norm" is used in each context:

### Norm in Data Scaling

![](standard_vs_norm.png)

In data scaling, "norm" typically refers to the process of normalizing data. This involves adjusting the values in the dataset so that they fit within a certain range, such as 0 to 1 or -1 to 1. This is usually done to ensure that all input features (variables) contribute equally to the analysis and to help algorithms perform better and converge faster. The two common types of norms used in data scaling are:
- **Min-Max Normalization**: Rescales data to a fixed range, usually 0 to 1.
- **Z-score Normalization (Standardization)**: Rescales data to mean of 0 and a standard deviation of 1.

### Norm in Layer Normalization

![](layernorm_demo_v1.png)

Layer Normalization is a technique used in training deep neural networks. It involves normalizing the inputs across the features for each data sample. This can be particularly useful in stabilizing the hidden state dynamics in recurrent networks, and it is also widely used in transformers and other types of networks. The "norm" in Layer Normalization refers to the process of calculating the mean and standard deviation across each entire layer's outputs for a single training case, and then using these statistics to normalize the outputs of the layer.

- **Process**: For each data sample in a batch, Layer Normalization computes the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Unlike Batch Normalization, which does this across the batch dimension, Layer Normalization does it across the features.

### Key Differences:
- **Scope**: In data scaling, normalization is typically applied to the entire dataset before training to adjust the range of data features. In contrast, Layer Normalization is applied to the outputs of individual layers within a neural network during training.
- **Purpose**: Data scaling normalization aims to adjust the scale of data for better convergence and performance of machine learning algorithms. Layer Normalization aims to stabilize learning and reduce the training time in deep neural networks by reducing internal covariate shift.
- **Application Level**: Normalization in data scaling is a preprocessing step, external to any model, while Layer Normalization is an integral part of model architecture and affects the learning process directly.

Thus, even though both use the term "norm," they apply it in different contexts and for different purposes within data science and machine learning.

### Will the kurtosis be preserved after Layernorm?

The effect of Layer Normalization on the kurtosis of a dataset can vary, but it typically does not preserve kurtosis. Layer Normalization (or any normalization that scales data based on its mean and variance) can significantly alter the shape of the distribution, including its tails and peak, which are key factors in determining kurtosis.

### Understanding Kurtosis:
Kurtosis measures the "tailedness" of a probability distribution:
- A **positive kurtosis** indicates a distribution with heavy tails and a sharp peak (leptokurtic).
- A **negative kurtosis** indicates a distribution with light tails and a flatter peak (platykurtic).
- A **kurtosis of zero** is associated with a normal distribution (mesokurtic).

### Effect of Layer Normalization:

![](layernorm_demo_v2.png)

Layer Normalization standardizes data by subtracting the mean and dividing by the standard deviation for each individual example in the dataset. This process reshapes the distribution of the data to have a mean of zero and a standard deviation of one. Here's how it can affect kurtosis:
- **Centralizing the Mean**: This step doesn't directly affect kurtosis, as it simply shifts the distribution.
- **Scaling by the Standard Deviation**: This changes the spread of the distribution. If the original data had extreme values or outliers, scaling would reduce the relative extremity of these points, potentially reducing the kurtosis if the original distribution was leptokurtic.

The degree to which kurtosis is changed depends on the original distribution of the data. If the data were uniformly distributed or normally distributed, the change in kurtosis might be less pronounced. However, for distributions with extreme values or outliers, the change can be significant.

To accurately determine the impact of layer normalization on kurtosis, it's beneficial to perform empirical tests or simulations to see how the distribution shape changes with different types of input data. Layer Normalization may not inherently preserve characteristics like kurtosis because its primary goal is to stabilize the learning process rather than maintain specific statistical properties of the input data.


## What is Gumbel-Softmax trick?

![](gumbel.png)
The Gumbel-Softmax trick is a method for sampling from a categorical distribution using a differentiable approximation, which allows for gradient-based optimization where standard categorical sampling cannot be used due to its non-differentiable nature. This method is particularly useful in training neural networks, especially in scenarios where you need to backpropagate through discrete variables.

### Understanding the Gumbel-Softmax Trick

The Gumbel-Softmax trick involves two main concepts: the Gumbel distribution and the Softmax function. Here's how it works:

1. **Gumbel Distribution**:
   - To sample from a categorical distribution with class probabilities \( p_1, p_2, \ldots, p_K \), you first need to introduce a way to convert these probabilities into a sample from the categorical distribution. 
   - For each class \( i \), you generate a Gumbel random variable \( G_i \) which can be obtained by transforming uniform random variables. Specifically, \( G_i \) can be sampled using:
     \[
     G_i = -\log(-\log(U_i))
     \]
     where \( U_i \) is a uniform random variable from 0 to 1. This transformation ensures that \( G_i \) follows a Gumbel distribution.

2. **Logits and Noise Addition**:
   - You compute the logits for each class, which are the unnormalized log probabilities \( \log(p_i) \). 
   - The Gumbel random variables are added to these logits. This step essentially perturbs the log probabilities with noise that has a specific extreme value distribution (Gumbel).

3. **Softmax Function**:
   - The perturbed logits are then passed through a softmax function, which is a differentiable function commonly used in multi-class classification problems in neural networks. The softmax function is given by:
     \[
     y_i = \frac{\exp((\log(p_i) + G_i) / \tau)}{\sum_{j=1}^K \exp((\log(p_j) + G_j) / \tau)}
     \]
   - Here, \( \tau \) is a temperature parameter that controls how closely the Gumbel-Softmax distribution approximates the categorical distribution. As \( \tau \) approaches 0, the samples become one-hot encoded (mimicking exact categorical samples), making the distribution more discrete. As \( \tau \) increases, the distribution becomes smoother and more continuous.

### Applications and Importance

The Gumbel-Softmax trick is particularly important in scenarios where:
- **End-to-End Learning**: You need to learn policies or other components that involve discrete decisions in an end-to-end trainable system.
- **Gradient Backpropagation**: The model involves discrete choices, and you want to use standard gradient-based learning techniques, which require differentiability.

This approach is widely used in reinforcement learning, variational autoencoders (VAEs) for discrete latent variables, and other areas of machine learning where modeling and learning discrete distributions in a differentiable manner are crucial.
