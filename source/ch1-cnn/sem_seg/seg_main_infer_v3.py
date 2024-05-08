import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import logging
from unet import UNet
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path, device, num_classes):
    try:
        model = UNet(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])  # Load only the model weights
        else:
            model.load_state_dict(checkpoint)  # Assuming the file might be only weights
        model.to(device)
        model.eval()
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error(f"The model file {model_path} was not found.")
        exit()
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {str(e)}")
        exit()


def preprocess_image(image_path, target_size):
    try:
        transform = transforms.Compose([
            transforms.Resize(target_size),  # Resize to the input size expected by the model
            transforms.ToTensor()            # Convert image to tensor
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        return image
    except FileNotFoundError:
        logging.error(f"The image file {image_path} was not found.")
        exit()
    except Exception as e:
        logging.error(f"An error occurred while processing the image: {str(e)}")
        exit()

def segment_image(model, image, device, class_names):
    image = image.unsqueeze(0).to(device)  # Add batch dimension and transfer to device
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(1).squeeze(0).cpu().numpy()  # Remove batch dim and transfer predictions to CPU

    # Identify unique classes detected in the image
    unique_classes = np.unique(prediction)
    detected_classes = [class_names[class_index] for class_index in unique_classes]

    logging.info(f"Detected classes: {detected_classes}")
    return prediction, detected_classes

def display_segmentation(original_image, segmentation, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(segmentation)
    axes[1].set_title('Segmentation Output')
    axes[1].axis('off')
    plt.savefig(output_path)
    plt.show()

def main(args):
    class_names = [
        "Background", "Aeroplane", "Bicycle", "Bird", "Boat",
        "Bottle", "Bus", "Car", "Cat", "Chair",
        "Cow", "Dining table", "Dog", "Horse", "Motorbike",
        "Person", "Potted plant", "Sheep", "Sofa", "Train",
        "TV/monitor"
    ]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")

    model = load_model(args.model_path, device, args.num_classes)
    image = preprocess_image(args.image_path, (args.image_size, args.image_size))
    original_image = Image.open(args.image_path)  # Load original image for visualization

    segmentation, detected_classes = segment_image(model, image, device, class_names)
    display_segmentation(original_image, segmentation, args.output_path)

    print("Classes detected in the image:", detected_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentation Inference Script")
    parser.add_argument('--image_path', type=str, default="bike.jpg", help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='seg_best.pth',help='Path to the trained model file')
    parser.add_argument('--output_path', type=str, default='segmentation_result.png', help='Path to save the segmentation result image')
    parser.add_argument('--image_size', type=int, default=256, help='The size (height, width) to which the input image is resized')
    parser.add_argument('--num_classes', type=int, default=21, help='Number of classes in the model')
    parser.add_argument('--cpu', action='store_true', help='Force to run the model on CPU, even if CUDA is available')
    args = parser.parse_args()

    main(args)
