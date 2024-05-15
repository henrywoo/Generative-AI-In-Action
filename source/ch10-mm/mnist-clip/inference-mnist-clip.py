import torch
import matplotlib.pyplot as plt
from clip import CLIP
from dataset import MNIST
import argparse
import os

# Setup device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset
def load_dataset(data_dir, is_train=True):
    return MNIST(data_dir=data_dir, is_train=is_train)

# Load the CLIP model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model path {model_path} does not exist.")
    model = CLIP().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Display a model using hiq.vis.print_model (if available)
def display_model(model):
    try:
        from hiq.vis import print_model
        print_model(model)
    except ImportError:
        print("hiq.vis.print_model is not available.")

# Classify a single image
def classify_image(model, image, targets):
    logits = model(image.unsqueeze(0).to(DEVICE), targets.to(DEVICE))
    return logits

# Find top k similar images
def find_similar_images(model, image, other_images, k=5):
    img_emb = model.img_enc(image.unsqueeze(0).to(DEVICE))
    other_img_embs = model.img_enc(torch.stack(other_images, dim=0).to(DEVICE))
    logits = img_emb @ other_img_embs.T
    values, indices = logits[0].topk(k)
    return indices

# Main function to handle workflow
def main(data_dir, model_path, k, use_train_data, digit_labels):
    # Load dataset for classification and similarity checks
    train_dataset = load_dataset(data_dir, is_train=True)
    test_dataset = load_dataset(data_dir, is_train=False)

    # Select dataset based on the use_train_data flag for similarity search
    similarity_dataset = train_dataset if use_train_data else test_dataset

    model = load_model(model_path)
    display_model(model)

    # Classification using the first image from test dataset (usually you test on test data)
    image, label = test_dataset[0]
    print('Ground truth classification:', label)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("gt.png")
    plt.show()

    targets = torch.arange(0, 10)  # 10 classes
    logits = classify_image(model, image, targets)
    print(logits)
    print('CLIP classification:', logits.argmax(-1).item())

    # Image similarity search
    other_images = [similarity_dataset[i][0] for i in range(1, 101)]
    other_labels = [similarity_dataset[i][1] for i in range(1, 101)]
    similar_indices = find_similar_images(model, image, other_images, k)

    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(similar_indices):
        plt.subplot(1, 5, i + 1)
        plt.imshow(other_images[idx].permute(1, 2, 0))
        plt.title(other_labels[idx])
        plt.axis('off')
    plt.savefig("gt_simiar.png")
    plt.show()

    if digit_labels:
        create_digit_classifier(model, test_dataset, digit_labels)

def create_digit_classifier(model, dataset, digit_labels):
    # Initialize a list to store images for each digit
    images = []
    for label in digit_labels:
        for image, img_label in dataset:
            if img_label == label:
                images.append(image)
                break  # Stop searching after finding the first match for the label

    # Concatenate images along width to create a single row image
    images_concat = torch.cat(images, dim=2)

    # Display the result
    plt.imshow(images_concat.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("digit_classifier.png")
    plt.show()
    return images_concat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Image Classification and Similarity Demo")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to CLIP model checkpoint")
    parser.add_argument("--k", type=int, default=5, help="Number of similar images to find")
    parser.add_argument("--use_train_data", action='store_true',
                        help="Whether to use training data for similarity search")
    parser.add_argument("--digit_labels", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="List of digits for creating the classifier")
    args = parser.parse_args()

    main(args.data_dir, args.model_path, args.k, args.use_train_data, args.digit_labels)
