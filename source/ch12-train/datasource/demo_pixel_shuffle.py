import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize

# Load the astronaut image from skimage
image = data.astronaut()
# Resize the image to a smaller size for demonstration
image = resize(image, (128, 128), anti_aliasing=True)
plt.imshow(image)
plt.title("Original Image")
plt.show()

# Convert the image to a tensor and add a batch dimension
image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

# Define the upsampling factor
upsample_factor = 2

# Simulate the process where the image is downsampled and channels are increased
# Here, we will simulate an image with 4x the number of channels
image_tensor = image_tensor.repeat(1, upsample_factor ** 2, 1, 1)

# Apply PixelShuffle to upscale the image
pixel_shuffle = nn.PixelShuffle(upsample_factor)
upscaled_image_tensor = pixel_shuffle(image_tensor)

# Convert the tensor back to a numpy array for display
upscaled_image = upscaled_image_tensor.squeeze(0).permute(1, 2, 0).numpy()

# Display the upscaled image
plt.imshow(upscaled_image)
plt.title("Upscaled Image with PixelShuffle")
plt.show()
