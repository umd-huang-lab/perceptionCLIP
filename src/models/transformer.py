import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import kornia.augmentation as K
from open_clip import tokenize
import random
import src.templates as templates


def denormalize(image):
    return K.Denormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(
        image)


def normalize(image):
    return T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(
        image)


def generate_random_param(augmentation_type, include_org=False):
    if include_org:
        low = 0
    else:
        low = 1

    if augmentation_type in (
            'vflip', 'invert', 'grayscale', 'random_erase', 'elastic_transform', 'solarize',
            'bright', 'dark', 'blur'):
        param = torch.randint(low=low, high=2, size=(1,)).item()
    elif augmentation_type == 'rotation':
        param = torch.randint(low=low, high=4, size=(1,)).item()
    elif augmentation_type in ('snow', 'frost', 'fog', 'noise', 'jpeg'):
        param = torch.randint(low=low, high=6, size=(1,)).item()
    else:
        raise Exception(f'Unknown augmentation type {augmentation_type}')

    return param


def augment_image(augmentation_type, image, param=None):
    if param is None:
        param = generate_random_param(augmentation_type)

    if augmentation_type == 'vflip':
        if param == 0:
            image = image
        elif param == 1:
            image = TF.vflip(image)
    elif augmentation_type == 'rotation':
        if param == 0:
            image = image
        elif param == 1:
            image = TF.rotate(image, 90)
        elif param == 2:
            image = TF.rotate(image, 180)
        elif param == 3:
            image = TF.rotate(image, 270)
    elif augmentation_type == 'random_erase':
        if param == 0:
            image = image
        elif param == 1:
            image = T.RandomErasing(p=1)(image)
    elif augmentation_type == 'elastic_transform':
        if param == 0:
            image = image
        elif param == 1:
            image = T.ElasticTransform(alpha=100.)(image)
    elif augmentation_type == 'invert':
        if param == 0:
            image = image
        elif param == 1:
            image = normalize(TF.invert(denormalize(image)))
    elif augmentation_type == 'solarize':
        if param == 0:
            image = image
        elif param == 1:
            image = normalize(TF.solarize(denormalize(image), 0.6))
    elif augmentation_type == 'grayscale':
        if param == 0:
            image = image
        elif param == 1:
            image = normalize(TF.rgb_to_grayscale(denormalize(image), num_output_channels=3))
    elif augmentation_type == 'bright':
        if param == 0:
            image = image
        elif param == 1:
            image = normalize(TF.adjust_brightness(denormalize(image), brightness_factor=2))
    elif augmentation_type == 'dark':
        if param == 0:
            image = image
        elif param == 1:
            image = normalize(TF.adjust_brightness(denormalize(image), brightness_factor=0.5))
    elif augmentation_type == 'blur':
        if param == 0:
            image = image
        elif param == 1:
            image = TF.gaussian_blur(image, 17)
    elif augmentation_type == 'snow':
        return image
    elif augmentation_type == 'frost':
        return image
    elif augmentation_type == 'fog':
        return image
    elif augmentation_type == 'noise':
        return image
    elif augmentation_type == 'jpeg':
        return image

    else:
        raise Exception(f'Unknown augmentation type {augmentation_type}')

    return image
