from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
from PIL import Image


class StanfordDogsDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.breeds = sorted(os.listdir(annotations_dir))

        # Parse annotations
        for breed_id, breed_dir in enumerate(self.breeds):
            breed_path = os.path.join(annotations_dir, breed_dir)
            if os.path.isdir(breed_path):
                for annotation_file in os.listdir(breed_path):
                    annotation_path = os.path.join(breed_path, annotation_file)
                    tree = ET.parse(annotation_path)
                    root = tree.getroot()
                    image_file = root.find('filename').text + ".jpg"
                    if image_file.endswith("%s.jpg"):
                        tmp = self.images_dir + "/" + breed_dir + "/" + annotation_file + ".jpg"
                        if os.path.isfile(tmp):
                            image_path = tmp
                    else:
                        image_path = os.path.join(self.images_dir, breed_dir, image_file)
                    self.image_paths.append(image_path)
                    self.labels.append(breed_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label