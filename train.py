import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import math
import os

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, einsum

import torch.nn.functional as F
from scipy.ndimage import rotate, zoom
from torch.autograd import Variable
import cv2

import random
from functools import reduce
from operator import mul

from augmentor import vit_transforms, intensity_augment, spatial_augment
import torchio as tio


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from dataset import CustomDataset

def train():


    # train_path = '/workspace/data/SAG_3D_DESS_v2_full/train.csv'
    # test_path = '/workspace/data/SAG_3D_DESS_v2_full/test.csv'
    # val_path = '/workspace/data/SAG_3D_DESS_v2_full/validation.csv'
    train_path = 'D:\lpmk\GSOFT\\vit-supcon\workspace\data\SAG_3D_DESS_v2\\train.csv'

    train_ds = CustomDataset(
        train_path, 
        vit_transforms=vit_transforms,
        contrastive_transforms=tio.Compose([tio.RescaleIntensity(out_min_max=(0, 1)),]))

    # val_ds = CustomDataset(
    #     val_path,
    #     vit_transforms=tio.Compose([tio.RescaleIntensity(out_min_max=(0, 1))]),
    #     contrastive_transforms=tio.Compose([tio.RescaleIntensity(out_min_max=(0, 1)),])
    # )

    # test_ds = CustomDataset(
    #     test_path, 
    #     vit_transforms=tio.Compose([tio.RescaleIntensity(out_min_max=(0,1))]),
    #     contrastive_transforms=tio.Compose([tio.RescaleIntensity(out_min_max=(0, 1)),])
    # )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    # test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    for batch in train_loader:
        vit_images = batch['vit']  
        contrastive_images = batch['contrastive']  
        mri_sample_1 = contrastive_images
        mri_sample_2 = vit_images
        labels = batch['label']
        print(vit_images.shape)
        print(contrastive_images.shape)
        print(labels)
        break


if __name__ == '__main__':
    train()