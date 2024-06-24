# Diffusion Model

Diffusion models are a class of generative models that have gained significant attention due to their ability to produce high-quality images. These models work by gradually transforming noise into a desired image through a series of steps. The core idea is to reverse a diffusion process that slowly adds noise to data, thereby generating data from noise. There are several types of diffusion models including Denoising Diffusion Probabilistic Models (DDPM), Denoising Diffusion Implicit Models (DDIM) and Latent Diffusion Models (LDM).

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

## CFG

![](cfg.webp)

一文解释 Diffusion Model (一) DDPM 理论推导

- https://zhuanlan.zhihu.com/p/565901160
- https://zhuanlan.zhihu.com/p/589106222 (https://zhuanlan.zhihu.com/p/594007789)
- https://fanpu.io/blog/2023/score-based-diffusion-models/

