import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset

class TestCSVImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['image_path']
        true_label = int(row['true_label'])
        pred_label = int(row['predicted_label'])

        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, filename, true_label, pred_label
