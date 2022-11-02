import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from tqdm import tqdm # DELETE
import pickle

class BuildingDataset(Dataset):
    def __init__(self, annotations_file, img_dir, load_from = None, save = True, transform_pre=None, transform_aug=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform_pre = transform_pre
        self.transform_aug = transform_aug
        self.load_from = load_from
        self.save = save

        if self.load_from is not None and os.path.exists(self.load_from):
            print('loading cached dataset...')
            self.load_images(self.load_from)
        else:
            print('pre-process dataset...')
            self.preprocess_images(self.img_dir)
            if self.save:
                print('saving to cache...')
                self.save_images(self.load_from)

    def preprocess_images(self, img_dir):

        self.images = []

        # read and pre-process
        for idx in tqdm(range(len(self.img_labels))):
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

    def save_images(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.images, f)

    def load_images(self, path):
        with open(path, 'rb') as f:
            self.images = pickle.load(f)
