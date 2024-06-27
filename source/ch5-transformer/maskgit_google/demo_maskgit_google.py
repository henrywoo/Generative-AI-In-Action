import jax
import os
import itertools
import urllib.request
from timeit import default_timer as timer
import argparse

from maskgit.utils import visualize_images, read_image_from_url, draw_image_with_bbox, Bbox
from maskgit.inference import ImageNet_class_conditional_generator


def create_checkpoints_directory():
    os.makedirs('checkpoints/', exist_ok=True)


def checkpoint_canonical_path(type_, resolution):
    return f'checkpoints/{type_}_imagenet{resolution}_checkpoint'


def download_checkpoints(models_to_download):
    for type_, resolution in models_to_download:
        canonical_path = checkpoint_canonical_path(type_, resolution)
        if os.path.isfile(canonical_path):
            print(f"Checkpoint for {resolution} {type_} already exists, not downloading again")
        else:
            source_url = f'https://storage.googleapis.com/maskgit-public/checkpoints/{type_}_imagenet{resolution}_checkpoint'
            print(f'Downloading {source_url} to {canonical_path}')
            urllib.request.urlretrieve(source_url, canonical_path)


def generate_images(generator, input_tokens, sample_rng, run_mode, image_size):
    if run_mode == 'normal':
        results = generator.generate_samples(input_tokens, sample_rng)
    elif run_mode == 'pmap':
        sample_rngs = jax.random.split(sample_rng, jax.local_device_count())
        results = generator.p_generate_samples()(input_tokens, sample_rngs)
        results = jax.device_get(results)
        print(type(results))
        results = results.reshape((-1, image_size, image_size, 3))
    return results


def main(args):
    create_checkpoints_directory()
    models_to_download = itertools.product(["maskgit", "tokenizer"], [256, 512])
    download_checkpoints(models_to_download)

    generator_256 = ImageNet_class_conditional_generator(image_size=256)
    generator_512 = ImageNet_class_conditional_generator(image_size=512)
    rng = jax.random.PRNGKey(args.seed)

    category = args.category_256
    label = int(category.split(')')[0])

    input_tokens = generator_256.create_input_tokens_normal(label)
    pmap_input_tokens = generator_256.pmap_input_tokens(input_tokens)

    rng, sample_rng = jax.random.split(rng)
    start_timer = timer()

    results = generate_images(generator_256, input_tokens, sample_rng, args.run_mode, 256)

    end_timer = timer()
    print(f"Generated {generator_256.eval_batch_size()} images in {end_timer - start_timer} seconds")

    visualize_images(results, title=f'results')

    category = args.category_512
    label = int(category.split(')')[0])
    bbox = Bbox(args.bbox)

    image = read_image_from_url(args.img_url, height=512, width=512)
    draw_image_with_bbox(image, bbox)

    latent_mask, input_tokens = generator_512.create_latent_mask_and_input_tokens_for_image_editing(image, bbox, label)
    pmap_input_tokens = generator_512.pmap_input_tokens(input_tokens)
    rng, sample_rng = jax.random.split(rng)

    if args.run_mode == 'normal':
        results = generator_512.generate_samples(input_tokens, sample_rng, start_iter=2, num_iterations=12)
    elif args.run_mode == 'pmap':
        sample_rngs = jax.random.split(sample_rng, jax.local_device_count())
        results = generator_512.p_edit_samples()(pmap_input_tokens, sample_rngs)
        results = results.reshape([-1, 512, 512, 3])

    composite_images = generator_512.composite_outputs(image, latent_mask, results)
    visualize_images(composite_images, title=f'outputs')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskGIT Image Generation")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--run_mode', type=str, default='normal', choices=['normal', 'pmap'],
                        help='Run mode: normal or pmap')
    parser.add_argument('--category_256', type=str, default='90) lorikeet',
                        help='Category for 256x256 image generation')
    parser.add_argument('--category_512', type=str, default='345) ox', help='Category for 512x512 image editing')
    parser.add_argument('--bbox', type=str, default='128_64_256_288', help='Bounding box for image editing')
    parser.add_argument('--img_url', type=str,
                        default='https://storage.googleapis.com/maskgit-public/imgs/class_cond_input_1.png',
                        help='URL of the input image for editing')

    args = parser.parse_args()
    main(args)
