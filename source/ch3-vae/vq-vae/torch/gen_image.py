import torch
import torch.nn.functional as F


def generate_images(pixelcnn, vqvae, device, num_samples=10):
    samples = torch.zeros(num_samples, 28, 28).long().to(device)  # Start with all zeros
    pixelcnn.eval()
    with torch.no_grad():
        for i in range(28):
            for j in range(28):
                out = pixelcnn(samples)  # Output is logits for next pixel value
                probs = F.softmax(out[:, :, i, j], dim=-1)
                samples[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)

        # Get embeddings for samples
        embeddings = vqvae.vq.embedding(samples.view(-1))

        # Reshape the embeddings to match the expected input of the decoder
        embeddings = embeddings.view(num_samples, -1, 28, 28)

        # Pass through the decoder
        decoded_samples = vqvae.decoder(embeddings)

    return decoded_samples.cpu().numpy()


