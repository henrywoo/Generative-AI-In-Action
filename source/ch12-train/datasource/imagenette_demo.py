import numpy as np
import matplotlib.pyplot as plt


IN_MEAN = np.array([0.485, 0.456, 0.406])
IN_STD = np.array([0.229, 0.224, 0.225])
def get_data(path="frgfm/imagenette",
             name="full_size",
             batch_size=4,
             image_size=224,
             split='train',
             shuffle=True,
             num_workers=1,
             transform=None,
             return_loader=False):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from datasets import load_dataset
    from PIL import Image
    from io import BytesIO

    dataset = load_dataset(path, name, split=split)
    if transform is None:
        resize_transform = transforms.Resize((image_size, image_size))
        to_tensor_transform = transforms.ToTensor()
        normalize_transform = transforms.Normalize(mean=IN_MEAN, std=IN_STD)

    def transform_function(imgs_):
        transformed_images = []
        for img in imgs_['image']:
            if 'transparency' in img.info:
                img = img.convert('RGBA')
            elif img.mode in ('1', 'P', 'I'):
                img = img.convert('RGB')
            with BytesIO() as output:
                img.save(output, format='JPEG')
                img_bytes = output.getvalue()
            m = Image.open(BytesIO(img_bytes))
            if transform is None:
                m = resize_transform(m)
                m = to_tensor_transform(m)
                m = normalize_transform(m)
            else:
                m = transform(m)
            transformed_images.append(m)
        imgs_['image'] = transformed_images
        return imgs_

    dataset.set_transform(transform_function)
    if return_loader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        return dataset


dataloader = get_data(return_loader=True, shuffle=False)
batch = next(iter(dataloader))
images, labels = batch['image'], batch['label']

IMAGENETTE_NAMES = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn',
                    'garbage truck', 'gas pump', 'golf ball', 'parachute']
fig, axes = plt.subplots(1, 4, figsize=(9, 2.4))
for i in range(4):
    image = images[i].permute(1, 2, 0).cpu().numpy()
    image = image * IN_STD+ IN_MEAN  # Denormalize
    image = np.clip(image, 0, 1)  # Clip values to be in the range [0, 1]
    label = labels[i].item()
    class_name = IMAGENETTE_NAMES[label]
    axes[i].imshow(image)
    axes[i].set_title(f"Label: {class_name}", fontsize=8)
    axes[i].axis("off")
plt.savefig('imagenette_demo.png')
plt.show()
