import numpy as np
import jax
import jax.numpy as jnp
import os
import itertools
from timeit import default_timer as timer

import maskgit
from maskgit.utils import visualize_images, read_image_from_url, restore_from_path, draw_image_with_bbox, Bbox
from maskgit.inference import ImageNet_class_conditional_generator

import itertools
import os
import urllib.request

# Create directory for checkpoints
os.makedirs('checkpoints/', exist_ok=True)

# Define the models to download
models_to_download = itertools.product(["maskgit", "tokenizer"], [256, 512])

# Function to simulate the checkpoint_canonical_path method
def checkpoint_canonical_path(type_, resolution):
    return f'checkpoints/{type_}_imagenet{resolution}_checkpoint'

# Iterate through the combinations and download the checkpoints if they don't exist
for (type_, resolution) in models_to_download:
    canonical_path = checkpoint_canonical_path(type_, resolution)
    if os.path.isfile(canonical_path):
        print(f"Checkpoint for {resolution} {type_} already exists, not downloading again")
    else:
        source_url = f'https://storage.googleapis.com/maskgit-public/checkpoints/{type_}_imagenet{resolution}_checkpoint'
        print(f'Downloading {source_url} to {canonical_path}')
        urllib.request.urlretrieve(source_url, canonical_path)

generator_256 = ImageNet_class_conditional_generator(image_size=256)
generator_512 = ImageNet_class_conditional_generator(image_size=512)
arbitrary_seed = 42
rng = jax.random.PRNGKey(arbitrary_seed)

run_mode = 'normal'  #@param ['normal', 'pmap']

p_generate_256_samples = generator_256.p_generate_samples()
p_edit_512_samples = generator_512.p_edit_samples()
category = "90) lorikeet"

label = int(category.split(')')[0])

# prep the input tokens based on the chosen label
input_tokens = generator_256.create_input_tokens_normal(label)
pmap_input_tokens = generator_256.pmap_input_tokens(input_tokens)

# we default to 256 here which is faster
image_size = 256

# NOTE that in both run modes, subsequent re-runs tend to be much faster
# than the initial run.

rng, sample_rng = jax.random.split(rng)
start_timer = timer()

# In "normal" mode, a batch of 8 images takes a V80
# ~25 seconds in 256x256, and ~75 seconds in 512x512.
if run_mode == 'normal':
    results = generator_256.generate_samples(input_tokens, sample_rng)
elif run_mode == 'pmap':
    sample_rngs = jax.random.split(sample_rng, jax.local_device_count())
    results = p_generate_256_samples(pmap_input_tokens, sample_rngs)

    # flatten the pmap results
    results = results.reshape([-1, image_size, image_size, 3])

end_timer = timer()
print(f"generated {generator_256.eval_batch_size()} images in {end_timer - start_timer} seconds")

# Visualize
visualize_images(results, title=f'results')

category = "345) ox"
label = int(category.split(')')[0])


# we switch to 512 here for demo purposes
image_size = 512

# Feel free to change the input below to your favorite example!
bbox_top_left_height_width = '128_64_256_288' # @param
img_url = 'https://storage.googleapis.com/maskgit-public/imgs/class_cond_input_1.png' # @param

bbox = Bbox(bbox_top_left_height_width)

# Load the input image, and visualize it with our bounding box
image = read_image_from_url(
    img_url,
    height=image_size,
    width=image_size)

draw_image_with_bbox(image, bbox)

latent_mask, input_tokens = generator_512.create_latent_mask_and_input_tokens_for_image_editing(
    image, bbox, label)

pmap_input_tokens = generator_512.pmap_input_tokens(input_tokens)
rng, sample_rng = jax.random.split(rng)

if run_mode == 'normal':
    # starting from [2] to represent the fact that we
    # already know some tokens from the given image
    results = generator_512.generate_samples(
        input_tokens,
        sample_rng,
        start_iter=2,
        num_iterations=12
        )

elif run_mode == 'pmap':
    sample_rngs = jax.random.split(sample_rng, jax.local_device_count())
    results = p_edit_512_samples(pmap_input_tokens, sample_rngs)
    # flatten the pmap results
    results = results.reshape([-1, image_size, image_size, 3])

#-----------------------
# Post-process by applying a gaussian blur using the input
# and output images.
composite_images = generator_512.composite_outputs(image, latent_mask, results)

#-----------------------
visualize_images(composite_images, title=f'outputs')