# Autoencoder

### How to Reconstruct an Image?

The facial image above can be described with various attributes, such as smiling, dark skin, male, bearded, not wearing glasses, and black hair. To reconstruct this facial image using a neural network architecture, you can first encode the image into different attributes and then decode these attributes to obtain the reconstructed image. This is the conventional approach of an autoencoder.

![](latent.webp)

However, there's a problem to solve here: how to encode the attributes of an image. Attributes can be described either as discrete values or as probability distributions (sampling discrete values from the probability distribution). For instance, the smile attribute of the first image of a little boy can be represented as a discrete value, e.g., -0.8, or as a probability distribution, e.g., a normal distribution ranging from -1 to 0 (then sampling a discrete value from it, most likely around the x-coordinate of the peak of the normal distribution, which is -0.5).


![](dist.png)

### AutoEncoder

The most basic AutoEncoder consists of an Encoder and a Decoder. The Encoder encodes the input image to obtain a latent code, which is then used by the Decoder to reconstruct the image. The reconstruction error between the input image and the generated image is calculated, and during training, this reconstruction error is minimized.

An AutoEncoder can be understood as learning any probability distribution through the network, and then using the x-coordinate of the highest point of this probability distribution as the discrete value of the encoding. This leads to the generation process of the AutoEncoder being uncontrollable and sensitive to input noise, as the learned probability distribution cannot be known in advance.


## Variational AutoEncoder (VAE)

![](vae.jpg)

**Variational Autoencoders (VAEs): A Bayesian Approach to Representation Learning**

VAEs are a powerful tool for learning meaningful, compressed representations (latent variables) from complex data, such as images.  They combine the strengths of neural networks with the principles of Bayesian inference.

In a VAE, the Encoder learns two encodings: the mean and the standard deviation. It then randomly samples a code from a normal distribution, and through the formula `sampled_code = epsilon * std + mean`, it resamples to get the latent code, which is then reconstructed by the Decoder.

Due to the continuity of the normal distribution, there is no issue with differentiability. The reparameterization trick allows for the recovery of the latent code and enables gradient updates through the chain rule.

![](vae.webp)

A VAE can be understood as a network learning the mean and standard deviation encodings of each attribute's normal distribution. Then, using the mean, standard deviation, and a normal distribution `N(0,1)`, each attribute's normal distribution is recovered, and discrete values for each attribute are randomly sampled.

![](vae2.webp)

As shown, the advantage of VAE over a traditional AutoEncoder is that when different samples are input, the VAE can robustly reconstruct images for any given sample. The VAE's generation process is controllable and insensitive to input noise, as each attribute is known to follow a normal distribution in advance.

**The Bayesian Framework:**

* **Prior Belief (p(z)):** Before observing any data, we hold certain assumptions about the underlying structure. This is often modeled as a simple distribution, like a standard Gaussian, over the latent variables.

* **Data as Evidence (x):**  The data we observe serves as evidence that can be used to update our prior beliefs.

* **Posterior Belief (p(z|x)):**  This is the revised understanding of the latent variables after considering the observed data. In VAEs, we seek to approximate this intractable posterior distribution.

**VAE Structure:**

1. **Generative Model:** The VAE assumes that the observed data is generated from a set of latent variables through a probabilistic process.  This can be thought of as a decoder network that takes latent variables as input and generates the data.

2. **Approximate Posterior (q(z|x)):** Calculating the exact posterior distribution is often infeasible.  VAEs use a neural network (the encoder) to approximate this posterior. It takes the observed data as input and outputs parameters for a distribution over the latent variables.

3. **Evidence Lower Bound (ELBO):** The ELBO serves as the objective function for training VAEs.  It's a lower bound on the log-likelihood of the data and balances two key aspects:

```
ELBO = E[log p(x|z)] - DKL[q(z|x) || p(z)]
```

* **E[log p(x|z)] - Reconstruction:** This is the expected log-likelihood of the data `x` given the latent representation `z`. It measures how well the VAE can reconstruct the original data from the latent variables. It encourages the model to learn meaningful latent variables that capture the essential features of the data.

* **DKL[q(z|x) || p(z)] - Regularization:** This is the Kullback-Leibler (KL) divergence between the approximate posterior distribution `q(z|x)` (encoded by the encoder network) and the prior distribution `p(z)`. It acts as a regularizer, encouraging the learned latent space to be close to the chosen prior distribution (often a standard normal distribution).


**The Bayesian Connection:**

VAEs can be seen as performing approximate Bayesian inference. The prior distribution represents our initial beliefs, the likelihood is encoded in the generative model, and the approximate posterior is learned through optimization. Maximizing the ELBO helps us find the best approximation of the true posterior distribution.

**Key Takeaways:**

* VAEs are powerful for unsupervised learning of meaningful representations.
* They combine neural networks with Bayesian principles.
* The ELBO balances data reconstruction accuracy with the complexity of the latent space model.

### Clarification on VAE

In VAE, each training sample x is mapped to a distribution p(z|x) in the latent space. Different samples x_i yield different distributions p(z_i | x_i). The KL divergence loss term encourages these p(z_i | x_i) distributions to approximate a standard normal distribution N(0, I). Therefore, VAE does **not encode the latent space as multiple normal distributions for different attributes**. Instead, VAE maps each input sample to a distribution in the latent space and uses the KL divergence to ensure that these distributions are close to the standard normal distribution. This allows for more structured and smooth sampling in the latent space, facilitating better generation and interpolation of new samples.

## Vector Quantized Variational AutoEncoder (VQVAE)

In VQVAE, the Encoder learns intermediate encodings, which are then mapped to one of the K vectors in the codebook through nearest-neighbor search. The Decoder reconstructs the image from these latent codes.

Because nearest-neighbor search uses `argmax` to find the index in the codebook, it introduces a **non-differentiable problem**. VQVAE uses the **stop-gradient operation** to avoid this issue. This means that during the forward pass, gradients are stopped, and during the backward pass, the gradients from the decoder input are copied to the encoder output.

```python
quantize = input + (quantize - input).detach()
# During forward propagation, it works as usual.
# During backpropagation, the gradient for the detach() part is 0, and the gradients of quantize and input are the same.
# This effectively copies quantize to input.
```

**Key Differences Between VQVAE and VAE:**

1. **Discrete Values for Attributes**: VQVAE directly finds discrete values for each attribute by looking up the nearest neighbor in the codebook.
2. **Codebook**: By maintaining a codebook, VQVAE provides more controllable encoding ranges.
3. **Higher Quality Images**: VQVAE can generate larger and higher resolution images compared to VAE.

VQVAE's method of using a codebook to manage **discrete latent codes** lays the groundwork for later models like DALLE and VQGAN.

![](vqvae.png)

### Summary

AutoEncoder, VAE, and VQVAE can be unified by the different designs of the probability distributions of their latent codes. 

- **AutoEncoder**: Learns an arbitrary probability distribution through the network.
- **VAE**: Designed with a normal distribution.
- **VQVAE**: Uses a codebook with a discrete distribution.

In essence, the reconstruction concept of AutoEncoders is to use low-dimensional latent code distributions to represent high-dimensional data distributions. Both VAE and VQVAE aim to control the image generation process by designing the form of the latent code distribution.

## Reference

- [漫谈VAE和VQVAE，从连续分布到离散分布](https://zhuanlan.zhihu.com/p/388299884) 
- [自编码器AE、VAE、dVAE、VQ-VAE、VQ-VAE2](https://www.p-chao.com/2024-01-28/%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8ae%E3%80%81vae%E3%80%81dvae%E3%80%81vq-vae%E3%80%81vq-vae2/)
- [生成模型之VAE与VQ-VAE](https://blog.csdn.net/m0_56214772/article/details/129711670)
- [GAN 和 VAE 的本质区别是什么？为什么两者总是同时被提起？](https://www.zhihu.com/question/317623081/answer/1994238294)
- [DALL-E论文笔记](https://www.p-chao.com/2024-01-21/dall-e%e8%ae%ba%e6%96%87%e7%ac%94%e8%ae%b0/#dVAE)