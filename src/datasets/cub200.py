import os

from PIL import Image
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CUBDataset(datasets.ImageFolder):
    _normalization_stats = {'mean': (0.485, 0.456, 0.406),
                            'std': (0.229, 0.224, 0.225)}

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):

        img_root = os.path.join(root, 'images')

        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.redefine_class_to_idx()

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train

        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in
                         enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUBDataset, self).__getitem__(index)

        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target

    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split('.')[-1].replace('_', ' ')
            split_key = k.split(' ')
            if len(split_key) > 2:
                k = '-'.join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict


class CUB200:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 custom=False,
                 seed=0,
                 ):
        location = os.path.join(location, 'CUB_200_2011')
        self.train_dataset = CUBDataset(root=location, transform=preprocess, train=True)
        self.val_dataset = None
        self.test_dataset = CUBDataset(root=location, transform=preprocess, train=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        self.val_loader = None
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)
        self.classnames = ['Black-footed Albatross',
                           'Laysan Albatross',
                           'Sooty Albatross',
                           'Groove-billed Ani',
                           'Crested Auklet',
                           'Least Auklet',
                           'Parakeet Auklet',
                           'Rhinoceros Auklet',
                           'Brewer Blackbird',
                           'Red-winged Blackbird',
                           'Rusty Blackbird',
                           'Yellow-headed Blackbird',
                           'Bobolink',
                           'Indigo Bunting',
                           'Lazuli Bunting',
                           'Painted Bunting',
                           'Cardinal',
                           'Spotted Catbird',
                           'Gray Catbird',
                           'Yellow-breasted Chat',
                           'Eastern Towhee',
                           'Chuck-will Widow',
                           'Brandt Cormorant',
                           'Red-faced Cormorant',
                           'Pelagic Cormorant',
                           'Bronzed Cowbird',
                           'Shiny Cowbird',
                           'Brown Creeper',
                           'American Crow',
                           'Fish Crow',
                           'Black-billed Cuckoo',
                           'Mangrove Cuckoo',
                           'Yellow-billed Cuckoo',
                           'Gray-crowned-Rosy Finch',
                           'Purple Finch',
                           'Northern Flicker',
                           'Acadian Flycatcher',
                           'Great-Crested Flycatcher',
                           'Least Flycatcher',
                           'Olive-sided Flycatcher',
                           'Scissor-tailed Flycatcher',
                           'Vermilion Flycatcher',
                           'Yellow-bellied Flycatcher',
                           'Frigatebird',
                           'Northern Fulmar',
                           'Gadwall',
                           'American Goldfinch',
                           'European Goldfinch',
                           'Boat-tailed Grackle',
                           'Eared Grebe',
                           'Horned Grebe',
                           'Pied-billed Grebe',
                           'Western Grebe',
                           'Blue Grosbeak',
                           'Evening Grosbeak',
                           'Pine Grosbeak',
                           'Rose-breasted Grosbeak',
                           'Pigeon Guillemot',
                           'California Gull',
                           'Glaucous-winged Gull',
                           'Heermann Gull',
                           'Herring Gull',
                           'Ivory Gull',
                           'Ring-billed Gull',
                           'Slaty-backed Gull',
                           'Western Gull',
                           'Anna Hummingbird',
                           'Ruby-throated Hummingbird',
                           'Rufous Hummingbird',
                           'Green Violetear',
                           'Long-tailed Jaeger',
                           'Pomarine Jaeger',
                           'Blue Jay',
                           'Florida Jay',
                           'Green Jay',
                           'Dark-eyed Junco',
                           'Tropical Kingbird',
                           'Gray Kingbird',
                           'Belted Kingfisher',
                           'Green Kingfisher',
                           'Pied Kingfisher',
                           'Ringed Kingfisher',
                           'White-breasted Kingfisher',
                           'Red-legged Kittiwake',
                           'Horned Lark',
                           'Pacific Loon',
                           'Mallard',
                           'Western Meadowlark',
                           'Hooded Merganser',
                           'Red-breasted Merganser',
                           'Mockingbird',
                           'Nighthawk',
                           'Clark Nutcracker',
                           'White-breasted Nuthatch',
                           'Baltimore Oriole',
                           'Hooded Oriole',
                           'Orchard Oriole',
                           'Scott Oriole',
                           'Ovenbird',
                           'Brown Pelican',
                           'White Pelican',
                           'Western-Wood Pewee',
                           'Sayornis',
                           'American Pipit',
                           'Whip-poor Will',
                           'Horned Puffin',
                           'Common Raven',
                           'White-necked Raven',
                           'American Redstart',
                           'Geococcyx',
                           'Loggerhead Shrike',
                           'Great-Grey Shrike',
                           'Baird Sparrow',
                           'Black-throated Sparrow',
                           'Brewer Sparrow',
                           'Chipping Sparrow',
                           'Clay-colored Sparrow',
                           'House Sparrow',
                           'Field Sparrow',
                           'Fox Sparrow',
                           'Grasshopper Sparrow',
                           'Harris Sparrow',
                           'Henslow Sparrow',
                           'Le-Conte Sparrow',
                           'Lincoln Sparrow',
                           'Nelson-Sharp-tailed Sparrow',
                           'Savannah Sparrow',
                           'Seaside Sparrow',
                           'Song Sparrow',
                           'Tree Sparrow',
                           'Vesper Sparrow',
                           'White-crowned Sparrow',
                           'White-throated Sparrow',
                           'Cape-Glossy Starling',
                           'Bank Swallow',
                           'Barn Swallow',
                           'Cliff Swallow',
                           'Tree Swallow',
                           'Scarlet Tanager',
                           'Summer Tanager',
                           'Artic Tern',
                           'Black Tern',
                           'Caspian Tern',
                           'Common Tern',
                           'Elegant Tern',
                           'Forsters Tern',
                           'Least Tern',
                           'Green-tailed Towhee',
                           'Brown Thrasher',
                           'Sage Thrasher',
                           'Black-capped Vireo',
                           'Blue-headed Vireo',
                           'Philadelphia Vireo',
                           'Red-eyed Vireo',
                           'Warbling Vireo',
                           'White-eyed Vireo',
                           'Yellow-throated Vireo',
                           'Bay-breasted Warbler',
                           'Black-and-white Warbler',
                           'Black-throated-Blue Warbler',
                           'Blue-winged Warbler',
                           'Canada Warbler',
                           'Cape-May Warbler',
                           'Cerulean Warbler',
                           'Chestnut-sided Warbler',
                           'Golden-winged Warbler',
                           'Hooded Warbler',
                           'Kentucky Warbler',
                           'Magnolia Warbler',
                           'Mourning Warbler',
                           'Myrtle Warbler',
                           'Nashville Warbler',
                           'Orange-crowned Warbler',
                           'Palm Warbler',
                           'Pine Warbler',
                           'Prairie Warbler',
                           'Prothonotary Warbler',
                           'Swainson Warbler',
                           'Tennessee Warbler',
                           'Wilson Warbler',
                           'Worm-eating Warbler',
                           'Yellow Warbler',
                           'Northern Waterthrush',
                           'Louisiana Waterthrush',
                           'Bohemian Waxwing',
                           'Cedar Waxwing',
                           'American-Three-toed Woodpecker',
                           'Pileated Woodpecker',
                           'Red-bellied Woodpecker',
                           'Red-cockaded Woodpecker',
                           'Red-headed Woodpecker',
                           'Downy Woodpecker',
                           'Bewick Wren',
                           'Cactus Wren',
                           'Carolina Wren',
                           'House Wren',
                           'Marsh Wren',
                           'Rock Wren',
                           'Winter Wren',
                           'Common Yellowthroat']
