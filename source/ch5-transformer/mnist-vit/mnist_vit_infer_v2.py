import matplotlib.pyplot as plt
import torch
from PIL import Image
from vit import ViT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备

def preprocess_image(image_path, plot=False):
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to the correct input size for your model
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel if your model expects grayscale input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust these values according to your model's training
    ])
    image = Image.open(image_path).convert('RGB')  # Open and convert image to RGB (then to grayscale in transforms)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    if plot:
        # Convert tensor to numpy for plotting
        image_array = image_tensor.squeeze(0).numpy()  # Remove batch dimension and convert to numpy
        image_array = np.clip(image_array * 0.229 + 0.485, 0, 1)  # Reverse normalization

        plt.imshow(image_array[0], cmap='gray')  # Plot the first channel of the image array
        plt.title("Processed Image")
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()

    return image_tensor

def main(image_path = '4.png'):
    model = ViT().to(DEVICE)
    model.load_state_dict(torch.load('model_best.pth'))
    model.eval()

    from hiq.vis import print_model
    print_model(model)
    # Load and preprocess the image
    image_tensor = preprocess_image(image_path, True)

    # Display the image
    plt.imshow(Image.open(image_path))
    plt.title('Input Image')
    plt.show()

    # Make a prediction
    with torch.no_grad():
        logits = model(image_tensor.to(DEVICE))
        predicted_class = logits.argmax(-1).item()
        print('Predicted classification:', predicted_class)


if __name__ == '__main__':
    main()
