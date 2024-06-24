# Diffusion Model

Diffusion models are a class of generative models that have gained significant attention due to their ability to produce high-quality images. These models work by gradually transforming noise into a desired image through a series of steps. The core idea is to reverse a diffusion process that slowly adds noise to data, thereby generating data from noise.

Mathematically, diffusion models aim to capture the complex patterns within high-dimensional data. Instead of directly estimating the probability distribution of the data p(x) like traditional likelihood-based models, diffusion models focus on predicting the gradient of the log probability, also known as the **score function**:

🔻x(log(p(x)))

$$
\nabla_x \log p(x)
$$

In python:

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

## DDPM

![](ddpm-pt-mnist-cond-unet/demo_diffusion_process.png)

[Code](vallina_ddpm_ddim/sampler/ddpm.py)

## DDIM

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

