import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
 '.npy'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

def npy_loader(path):
    return np.load(path)


class NumpyDataset_mask(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[256, 256], loader=npy_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs_img = transforms.Compose([
                transforms.ToTensor(),
        ])
        self.tfs_mask = transforms.Compose([
                transforms.ToTensor(),
        ])
        self.loader = npy_loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]) + '.npy'
        
        img = self.tfs_img(self.loader('{}/{}/{}'.format(self.data_root, 'Tmax', file_name)))
        cond_image = self.tfs_img(self.loader('{}/{}/{}'.format(self.data_root, 'mCTA', file_name)))
        mask = self.tfs_mask(self.loader('{}/{}/{}'.format(self.data_root, 'mask', file_name)))
        mask = mask + 0

        img[img >1]=1
        img[img <0]=0
        cond_image[cond_image >1]=1
        cond_image[cond_image <0]=0

        cond_image = cond_image *np.concatenate([mask,mask,mask,mask], axis=0)
        mask_img = img*(1. - mask) + mask
        img = img * mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = file_name
        return ret 

    def __len__(self):
        return len(self.flist)