from torch.utils.data import Dataset
import torchvision
from torchvision.transforms.v2 import PILToTensor, Compose

class MNIST(Dataset):
    def __init__(self, data_dir='./mnist/', is_train=True):
        super().__init__()
        self.ds = torchvision.datasets.MNIST(root=data_dir, train=is_train, download=True)
        self.img_convert = Compose([
            PILToTensor(),
            lambda x: x.float() / 255.0,
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        return self.img_convert(img), label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = MNIST()
    img, label = ds[0]
    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
