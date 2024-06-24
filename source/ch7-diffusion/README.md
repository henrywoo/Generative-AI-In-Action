# Diffusion Model

Diffusion models are a class of generative models that have gained significant attention due to their ability to produce high-quality images. These models work by gradually transforming noise into a desired image through a series of steps. The core idea is to reverse a diffusion process that slowly adds noise to data, thereby generating data from noise.

Mathematically, diffusion models aim to capture the complex patterns within high-dimensional data. Instead of directly estimating the probability distribution of the data p(x) like traditional likelihood-based models, diffusion models focus on predicting the gradient of the log probability, also known as the **score function**:

$$
\nabla_x \log p(x)
$$

In python, it is like this:

```python
import torch

def score_function(x, model):
    """
    Computes the score function for a given data point x using a neural network model.
    Args:
        x (torch.Tensor): Input data point.
        model (torch.nn.Module): Neural network model trained to predict the score function.
    Returns:
        torch.Tensor: Score function evaluated at x.
    """
    x.requires_grad_(True)  # Enable gradient computation for x
    log_prob = model(x)    # Get log probability from the model
    score = torch.autograd.grad(log_prob.sum(), x)[0]  # Compute gradient w.r.t. x
    return score
```

This score function provides information about the direction in which the probability density increases most rapidly at a given point in the data space. By learning to predict the score function, diffusion models can generate new samples by iteratively refining a random noise input based on the predicted gradient information.


There are several types of diffusion models including Denoising Diffusion Probabilistic Models (DDPM), Denoising Diffusion Implicit Models (DDIM) and Latent Diffusion Models (LDM).

**DDPM** is a foundational type of diffusion model introduced by Jonathan Ho et al. in 2020. It uses a Markovian process to iteratively denoise a sample, starting from pure noise. It works by reversing a gradual noising process. The model is trained to predict the added noise at each step. It generates high-quality images, albeit with a relatively slow generation process due to the many steps involved.

**DDIM** is an extension of DDPM that introduces a non-Markovian forward process. This approach allows for fewer steps in the reverse process, speeding up generation while maintaining image quality. It provides a deterministic mapping from noise to data and reduces the number of steps required for generation, making it faster than DDPM.

**LDMs** are a variation of diffusion models that operate in a latent space rather than the pixel space. Because it applies the diffusion process in this compressed space and reconstructs high-resolution images from the latent space efficiently, this approach significantly reduces computational complexity and speeds up the generation process.

**Stable Diffusion** is a specific implementation of latent diffusion models developed by Stability AI. It's designed to be highly efficient and capable of generating high-resolution images quickly.

- **Versions**:
  - **SD v1**: The initial release, providing a solid foundation for stable and efficient image generation.
  - **SD v1.5**: An improved version with better stability and image quality.
  - **SD v2.1**: Further enhancements in image quality and generation speed.
  - **SD XL**: A version designed for even higher resolution and fidelity.
  - **SD 3**: The latest iteration with state-of-the-art improvements in stability, speed, and image quality.


Diffusion models, including DDPM, DDIM, and LDM, represent an exciting direction in generative modeling. They are particularly effective at generating high-quality images by reversing a noise process. Implementations like Stable Diffusion have brought these techniques into practical use, enabling the creation of impressive visuals with efficient computation.

## Denoising Diffusion Probabilistic Models (DDPM)

Denoising Diffusion Probabilistic Models (DDPM) are a type of generative model that have gained attention for their ability to generate high-quality images. DDPMs are built on the concept of diffusion, a process where data is gradually transformed into a simpler form by adding noise, and then reconstructed back by removing the noise.

![](ddpm-paper-1.webp)

### Model Training (Left Loop):

1. Loop over the epochs.
2. Sample a batch of images from the dataset.
3. For each image in the batch, sample a value of t uniformly.
4. Add noise to each image using a Gaussian Distribution with mean 0 and unit variance.
5. The model predicts the noise in each image at the given timestep t.
6. Compute the Mean Squared Error (MSE) loss between the sampled noise and the predicted noise for each image.

Instead of modeling the entire diffusion process as a single process, we can model each individual timestep separately. This approach speeds up training and likely results in a more stable training setting. By sampling the value of t uniformly for each training image, the model learns to handle all values of t while also learning the real image distribution.

### Image Generation/Sampling (Right Loop):

1. Sample noise from a Gaussian Distribution with mean 0 and unit variance. This represents the noisy image at time T.
2. Loop from time t = T to t = 1:
   - Sample new noise from a Gaussian Distribution to move the image to the previous timestep, t-1.
   - Using the trained model ε_θ, predict the noise at the current timestep. Remove this noise to move the image to the previous timestep t-1.
3. Repeat the loop until t = 1.
4. After completing all T iterations, a new image will be generated at timestep 0.

> [Code](vallina_ddpm_ddim/sampler/ddpm.py)

![](ddpm-pt-mnist-cond-unet/demo_diffusion_process.png)

## Denoising Diffusion Implicit Models (DDIM)

DDIM is an extension and improvement of DDPM. DDIMs enhance the diffusion-based generative modeling framework by introducing a non-Markovian reverse diffusion process. This modification results in faster and higher-quality image generation while retaining the robustness of the training phase seen in DDPMs. By leveraging deterministic transformations during the reverse process, DDIMs provide an efficient and effective way to generate realistic images from noise.

### How DDIM Works?

#### Training Phase

There is no difference between DDPM (Denoising Diffusion Probabilistic Models) and DDIM (Denoising Diffusion Implicit Models) in the training stage. Both models are trained using the same objective: to predict the noise that was added to the original data.

#### Generation/Sampling Phase (Reverse Diffusion Process with DDIM)

The key difference between DDIM and DDPM lies in the reverse diffusion process. In the sampling process, DDPM focuses solely on predicting the noise at each timestep. It starts from pure noise and iteratively denoises it based on the predicted noise at each step, eventually arriving at the generated image. DDIM modifies the reverse diffusion to be **non-Markovian**, meaning that the generation process does not depend solely on the previous timestep but can incorporate a more flexible transformation. DDIM's sampling process, however, incorporates information about both **the predicted noise** and **the current state of the image at each timestep**. This allows DDIM to take more direct "steps" towards the final image, resulting in faster sampling. In other words, DDIM's sampling process can be seen as having a more "goal-oriented" approach compared to DDPM's more "noise-focused" approach. This allows it to generate high-quality samples in fewer steps.

1. **Initialize with Noise**: Start with noise sampled from a Gaussian Distribution with mean 0 and unit variance. This represents the noisy image at time T.
2. **Non-Markovian Reverse Process**:
   - Loop from time t = T to t = 1 with a more flexible step size.
   - Instead of using new noise for each timestep, DDIM uses a deterministic approach that transforms the noisy image at the current timestep to the previous timestep directly.
   - The transformation is guided by the trained model (denoted as ε_θ) to predict the noise and adjust the image accordingly.
3. **Repeat**: Continue this process until t = 1.
4. **Final Image**: After completing all T iterations, the final generated image at timestep 0 is obtained.

[Code](vallina_ddpm_ddim/sampler/ddim.py)

## Conditional Diffusion

### Text Condition

[Code](ddpm-pt-mnist-cond-unet/unet.py)

[Model Checkpoint](https://drive.google.com/file/d/1EPAEqMTVnOacVbZsaJcj2zajaQ9N99Ju/view?usp=drive_link)

Label: 0
Generated Image: 

![](ddpm-pt-mnist-cond-unet/output_images/generated_image_1.png)

## Classifier-Free Guidance (CFG)

Simply put, the CFG scale (classifier-free guidance scale) or guidance scale is a parameter that controls how much the image generation process follows the text prompt. The higher the value, the more the image sticks to a given text input. But this does not mean that the value should always be set to maximum, as more guidance means less diversity and quality.


![](cfg.webp)

- https://blog.easydiffusion.online/the-cfg-scale-in-stable-diffusion/

一文解释 Diffusion Model (一) DDPM 理论推导

- https://zhuanlan.zhihu.com/p/565901160
- https://zhuanlan.zhihu.com/p/589106222 (https://zhuanlan.zhihu.com/p/594007789)
- https://fanpu.io/blog/2023/score-based-diffusion-models/

