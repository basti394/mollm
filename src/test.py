import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        for folder_name in os.listdir(root):
            folder_path = os.path.join(root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label = int(folder_name)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = Image.open(image_path).convert('RGB')
                self.images.append(image)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
