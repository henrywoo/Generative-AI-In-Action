import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()  
        )

    def forward(self, image):
        return self.cnn(image)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)  

    def forward(self, text):
        embedded = self.embedding(text)  
        output, _ = self.lstm(embedded) 
        return output[-1, :, :]  

class MultimodalModel(nn.Module):
    def __init__(self, image_features, text_features, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(image_features + text_features, 512)
        self.fc2 = nn.Linear(512, output_dim)  # Adjust for your task

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        combined = torch.cat((image_features, text_features), dim=1) 
        output = self.fc1(combined)
        output = self.fc2(output)
        return output


if __name__ == '__main__':
    mm = MultimodalModel(1024, 1024, 10)
    from hiq.vis import print_model
    print_model(mm)