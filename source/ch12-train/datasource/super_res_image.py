import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# Let's download an image
url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
response = requests.get(url)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = low_res_img.resize((128, 128))


# Define a simple PixelShuffle layer
class SimplePixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(SimplePixelShuffle, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(x)


# Convert the image to a tensor
low_res_img_tensor = torch.from_numpy(np.array(low_res_img)).permute(2, 0, 1).unsqueeze(0).float()

# Simulate the process where the image is downsampled and channels are increased
upscale_factor = 2
low_res_img_tensor = low_res_img_tensor.repeat(1, upscale_factor ** 2, 1, 1)

# Apply PixelShuffle
pixel_shuffle = SimplePixelShuffle(upscale_factor)
pixel_shuffled_img_tensor = pixel_shuffle(low_res_img_tensor)

# Convert the pixel shuffled tensor back to an image
pixel_shuffled_img = pixel_shuffled_img_tensor.squeeze(0).permute(1, 2, 0).byte().numpy()
pixel_shuffled_img = Image.fromarray(pixel_shuffled_img)

# Upscale the image using the Stable Diffusion Upscale Pipeline
prompt = "a white cat"
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]

# Plot the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(low_res_img)
axes[0].set_title("Original Low-Res Cat")
axes[0].axis("off")

axes[1].imshow(pixel_shuffled_img)
axes[1].set_title("Pixel Shuffled Cat")
axes[1].axis("off")

axes[2].imshow(upscaled_image)
axes[2].set_title("Upscaled Cat")
axes[2].axis("off")
plt.savefig("super_res_image.png")
plt.show()
