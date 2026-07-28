import os
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class GTSRBDataset(Dataset):

    def __init__(self, csv_file, root_dir):

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],         # so it matches the format from ImageNet
                std=[0.229, 0.224, 0.225]
            )       
    ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = os.path.join(self.root_dir, row["Path"])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        label = row["ClassId"]

        image = self.transform(image)

        return image, label
    
     