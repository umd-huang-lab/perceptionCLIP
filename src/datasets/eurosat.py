import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class EuroSAT:

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 custom=False,
                 seed=0,
                 **kwargs):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = torchvision.datasets.EuroSAT(root=location, transform=preprocess)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)

        self.classnames = [
            'annual crop land',
            'forest',
            'brushland or shrubland',
            'highway or road',
            'industrial buildings or commercial buildings',
            'pasture land',
            'permanent crop land',
            'residential buildings or homes or apartments',
            'river',
            'lake or sea',
        ]

