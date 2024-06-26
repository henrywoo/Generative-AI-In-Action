import argparse
import itertools
import os
import urllib.request
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import numpy as np

import maskgit
from maskgit.utils import visualize_images, read_image_from_url, restore_from_path, draw_image_with_bbox, Bbox
from maskgit.inference import ImageNet_class_conditional_generator

def create_checkpoints_directory():
    os.makedirs('checkpoints/', exist_ok=True)

def checkpoint_canonical_path(type_, resolution):
    return f'checkpoints/{type_}_imagenet{resolution}_checkpoint'

def download_checkpoints(models_to_download):
    for (type_, resolution) in models_to_download:
        canonical_path = checkpoint_canonical_path(type_, resolution)
        if os.path.isfile(canonical_path):
            print(f"Checkpoint for {resolution} {type_} already exists, not downloading again")
        else:
            source_url = f'https://storage.googleapis.com/maskgit-public/checkpoints/{type_}_imagenet{resolution}_checkpoint'
            print(f'Downloading {source_url} to {canonical_path}')
            urllib.request.urlretrieve(source_url, canonical_path)

def initialize_generators():
    generator_256 = ImageNet_class_conditional_generator(image_size=256)
    generator_512 = ImageNet_class_conditional_generator(image_size=512)
    return generator_256, generator_512

def generate_images(generator, label, run_mode, rng):
    input_tokens = generator.create_input_tokens_normal(label)
    pmap_input_tokens = generator.pmap_input_tokens(input_tokens)
    image_size = generator.image_size

    rng, sample_rng = jax.random.split(rng)
    start_timer = timer()

    if run_mode == 'normal':
        results = generator.generate_samples(input_tokens, sample_rng)
    elif run_mode == 'pmap':
        sample_rngs = jax.random.split(sample_rng, jax.local_device_count())
        results = generator.p_generate_samples()(pmap_input_tokens, sample_rngs)
        results = results.reshape([-1, image_size, image_size, 3])

    end_timer = timer()
    print(f"Generated {generator.eval_batch_size()} images in {end_timer - start_timer} seconds")
    visualize_images(results, title='results')

def edit_image(generator, label, image_url, bbox_str, run_mode, rng):
    bbox = Bbox(bbox_str)
    image = read_image_from_url(image_url, height=generator.image_size, width=generator.image_size)
    draw_image_with_bbox(image, bbox)

    latent_mask, input_tokens = generator.create_latent_mask_and_input_tokens_for_image_editing(image, bbox, label)
    pmap_input_tokens = generator.pmap_input_tokens(input_tokens)
    rng, sample_rng = jax.random.split(rng)

    if run_mode == 'normal':
        results = generator.generate_samples(input_tokens, sample_rng, start_iter=2, num_iterations=12)
    elif run_mode == 'pmap':
        sample_rngs = jax.random.split(sample_rng, jax.local_device_count())
        results = generator.p_edit_samples()(pmap_input_tokens, sample_rngs)
        results = results.reshape([-1, generator.image_size, generator.image_size, 3])

    composite_images = generator.composite_outputs(image, latent_mask, results)
    visualize_images(composite_images, title='outputs')

def main(args):
    create_checkpoints_directory()

    models_to_download = itertools.product(["maskgit", "tokenizer"], [256, 512])
    download_checkpoints(models_to_download)

    generator_256, generator_512 = initialize_generators()
    rng = jax.random.PRNGKey(args.seed)

    if args.action == 'generate':
        generate_images(generator_256, args.label, args.run_mode, rng)
    elif args.action == 'edit':
        edit_image(generator_512, args.label, args.img_url, args.bbox, args.run_mode, rng)
    else:
        print("Invalid action specified. Use 'generate' or 'edit'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskGIT Image Generation and Editing")
    parser.add_argument('--action', type=str, required=True, choices=['generate', 'edit'], help="Action to perform: 'generate' or 'edit'")
    parser.add_argument('--label', type=int, required=True, help="Label for image generation or editing")
    parser.add_argument('--run_mode', type=str, default='normal', choices=['normal', 'pmap'], help="Run mode: 'normal' or 'pmap'")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--img_url', type=str, help="Image URL for editing")
    parser.add_argument('--bbox', type=str, help="Bounding box for image editing in 'top_left_height_width' format")

    args = parser.parse_args()
    main(args)
