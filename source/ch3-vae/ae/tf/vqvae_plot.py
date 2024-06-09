from config import *

def show_all_subplots(originals, reconstructions, fig_title="Original vs. Reconstructed"):
    num_images = len(originals)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 0.7, 2))

    for i in range(num_images):
        axes[0, i].imshow(originals[i].squeeze() + 0.5)
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructions[i].squeeze() + 0.5)
        axes[1, i].axis("off")

    fig.suptitle(fig_title, fontsize=10)
    plt.tight_layout()
    save_fig("vqvae_ori_recon")
    plt.show()


def plot_original_vs_code(test_images, codebook_indices, fig_title="Original vs. Code"):
    num_images = len(test_images)
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 0.7, 2))
    for i in range(num_images):
        axs[0, i].imshow(test_images[i].squeeze() + 0.5)
        axs[0, i].axis("off")
        axs[1, i].imshow(codebook_indices[i])
        axs[1, i].axis("off")
    fig.suptitle(fig_title, fontsize=10)
    plt.tight_layout()
    save_fig("vqvae_ori_code")
    plt.show()


def plot_code_vs_generated(priors, generated_samples, fig_title="Code vs. Generated"):
    num_images = len(priors)
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 0.7, 2))
    for i in range(num_images):
        axs[0, i].imshow(priors[i])
        axs[0, i].axis("off")
        axs[1, i].imshow(generated_samples[i].squeeze() + 0.5)
        axs[1, i].axis("off")
    fig.suptitle(fig_title, fontsize=10)
    plt.tight_layout()
    save_fig("vqvae_generated")
    plt.show()