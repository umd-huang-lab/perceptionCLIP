import glob
import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv


class WaterbirdsDataset(Dataset):
    _normalization_stats = {'mean': (0.485, 0.456, 0.406),
                            'std': (0.229, 0.224, 0.225)}

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.data_dir = self.root_dir

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df["y"].values

        # We only support one confounder for CUB for now
        self.attribute_array = self.metadata_df["place"].values

        # Map to groups
        self.n_groups = pow(2, 2)
        assert self.n_groups == 4, "check the code if you are running otherwise"
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.attribute_array).astype("int")

        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }
        ind_split = (self.split_array == self.split_dict[self.split])
        self.y_array = self.y_array[ind_split]
        self.filename_array = self.filename_array[ind_split]
        self.group_array = self.group_array[ind_split]
        self.attribute_array = self.attribute_array[ind_split]

        # Image transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transform_waterbirds()

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        a = self.attribute_array[idx]
        img_filename = os.path.join(self.data_dir,
                                    self.filename_array[idx])
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        return img, y, a


def get_transform_waterbirds():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=WaterbirdsDataset._normalization_stats['mean'],
                             std=WaterbirdsDataset._normalization_stats['mean']),
    ])
    return transform


class Waterbirds:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 custom=False,
                 seed=0,
                 ):
        location = os.path.join(location, 'waterbird_complete95_forest2water2')
        self.train_dataset = WaterbirdsDataset(root_dir=location, split='train')
        self.val_dataset = WaterbirdsDataset(root_dir=location, split='val')
        self.test_dataset = WaterbirdsDataset(root_dir=location, split='test')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)
        self.classnames = ['landbird', 'waterbird']
        self.attrnames = ['land', 'water']

