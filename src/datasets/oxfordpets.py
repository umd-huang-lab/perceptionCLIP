import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class OxfordPets:

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 custom=False,
                 seed=0,
                 **kwargs):
        self.train_dataset = torchvision.datasets.OxfordIIITPet(root=location, split='trainval',
                                                                transform=preprocess)
        self.val_dataset = None
        self.test_dataset = torchvision.datasets.OxfordIIITPet(root=location, split='test',
                                                               transform=preprocess)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        self.val_loader = None
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)

        self.classnames = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier',
                           'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer',
                           'British Shorthair', 'Chihuahua', 'Egyptian Mau',
                           'English Cocker Spaniel', 'English Setter', 'German Shorthaired',
                           'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger',
                           'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian',
                           'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard',
                           'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx',
                           'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']
