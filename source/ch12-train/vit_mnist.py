from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

# Load a pre-trained Vision Transformer
model_name = 'google/vit-base-patch16-224'
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values']

def predict(image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = outputs.logits.argmax(-1).item()
    return pred

# Example usage
image_path = '4.png'
image_tensor = prepare_image(image_path)
prediction = predict(image_tensor)
print(f'Predicted Class: {prediction}')
