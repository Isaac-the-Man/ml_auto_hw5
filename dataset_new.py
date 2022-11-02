import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class BuildingDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform_pre=None, transform_aug=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform_pre = transform_pre
        self.transform_aug = transform_aug

        self.preprocess_images(self.img_dir)

    def preprocess_images(self, img_dir):

        self.images = []
        
        for idx in range(len(self.img_labels)):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            image = self.transform_pre(image)

            self.images.append(image)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = self.images[idx]

        label = self.img_labels.iloc[idx, 1]

        if not self.transform_aug == None:
            image = self.transform_aug(image)
            
        return image, label
