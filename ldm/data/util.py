import os
import torch
import torchvision
import torch.nn.functional as F
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
rot = torchvision.transforms.RandomRotation((45, 235))
vflip = torchvision.transforms.RandomVerticalFlip()
color = torchvision.transforms.ColorJitter(.25, .25, .25, .25)
dist = torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
elastic = torchvision.transforms.ElasticTransform(alpha=250.0)

def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]

    # Determine the amount of padding needed for tensor1 to match the size of tensor2
    LR_pad = (imgs[1].size(1) - imgs[0].size(1), imgs[1].size(2) - imgs[0].size(2))
    ref_pad = (imgs[1].size(1) - imgs[3].size(1), imgs[1].size(2) - imgs[3].size(2))

    # Pad tensor1
    LR_padded = F.pad(imgs[0], (0, LR_pad[1], 0, LR_pad[0]))
    ref_padded = F.pad(imgs[3], (0, ref_pad[1], 0, ref_pad[0]))

    if split == 'train':
        imgs = torch.stack([LR_padded, imgs[1], imgs[2], ref_padded], dim=0)
        # imgs = torch.stack(imgs, 0)
        if random.random() < .5:
            imgs = hflip(imgs)
            if random.random() < .25:
                imgs = rot(imgs)
                imgs = vflip(imgs)
                # imgs = elastic(imgs)
                # imgs = color(imgs)
                if random.random() < .15:
                    imgs = dist(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img