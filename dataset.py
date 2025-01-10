import torch
import numpy as np
import pandas as pd
import math
import os
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, df_path, vit_transforms=None, contrastive_transforms=None, flip_prob=0.5, num_classes = 5):
        self.df = pd.read_csv(df_path)
        self.vit_transforms = vit_transforms
        self.contrastive_transforms = contrastive_transforms
        self.flip_prob = flip_prob
        self.num_classes = num_classes

    def process_label(raw_label):
        if raw_label == 1:
            raw_label = 0
        if raw_label == 2 or raw_label == 3 or raw_label == 4:
            raw_label = 1
        return raw_label

    def random_flip(self, mri):
        if np.random.rand() < self.flip_prob:
            vit = np.flip(mri, axis=1).copy()
            cons = np.flip(mri, axis=1).copy()
        else:
            vit = mri.copy()
            cons = mri.copy()
        return vit, cons

        
    def __getitem__(self, index):
        mri_object = self._load_mri(index)
        vit_object, contrastive_object = self.random_flip(mri_object)
        
        vit_object = self._process_for_vit(vit_object)
        contrastive_object = self._process_for_contrastive(contrastive_object)
        
        label = self.df.loc[index]['kl_grade']
        if self.num_classes == 2:
            label = self.process_label(label)
        
        return {
            'vit': vit_object,
            'contrastive': contrastive_object,
            'label': label
        }
    
    def _load_mri(self, index):
        path_object = self.df.loc[index]['mri_path']
        # mri_file = '/workspace/data/SAG_3D_DESS_v2_full/MRI_Numpy/' + path_object
        mri_file = os.path.join('D:\lpmk\GSOFT\\vit-supcon\workspace\data\SAG_3D_DESS_v2\MRI_Numpy', path_object)
        mri_dict = np.load(mri_file)
        mri_object = mri_dict['data']
        mri_object = np.expand_dims(mri_object, 0)
        return mri_object
        
    def _process_for_vit(self, object):
        if self.vit_transforms:
            vit_object = self.vit_transforms(object)
        return torch.tensor(vit_object, dtype=torch.float32)
    
    def _process_for_contrastive(self, object):
        if self.contrastive_transforms:
            contrastive_object = self.contrastive_transforms(object)
        return torch.tensor(contrastive_object, dtype=torch.float32)

    def __len__(self):
        return len(self.df)