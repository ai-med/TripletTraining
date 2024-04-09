# This file is part of From Barlow Twins to Triplet Training: Differentiating Dementia with Limited Data (Triplet Training).
#
# Triplet Training is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Triplet Training is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Triplet Training. If not, see <https://www.gnu.org/licenses/>.

import os
import numpy as np
import torch
from typing import Optional
import logging
import h5py
import monai.transforms as montrans
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchio as tio
from tqdm import tqdm

from src.addiagnosis.datamodules.data_transform import *

LOG = logging.getLogger(__name__)

def get_images_from_h5(file_path):
    images = []

    with h5py.File(file_path, 'r') as file:
        # Iterate through items in the root of the HDF5 file
        for name, item in file.items():
            # Check if the item is a group
            scan = np.array(item['MRI/T1/data'])
            images.append(scan)

    return images

def get_image_transform(is_training: bool, 
                        is_ssl: bool = False,
                        original_height: int = 120,
                        input_height: int = 55):
    
    if is_ssl:
        roi_size = np.random.randint(low = input_height, high = original_height)
        return SSLTransform(is_training = is_training,
                            original_height = original_height, 
                            input_height = input_height,
                            roi_size = roi_size)

    img_transforms = []
    if is_training:
        randomAffineWithRot = tio.RandomAffine(
            scales=0.05,
            degrees=90,  # +-90 degree in each dimension
            translation=8,  # +-8 pixels offset in each dimension.
            image_interpolation="linear",
            default_pad_value="otsu",
            p=0.5,
        )
        randomFlip = tio.RandomFlip(axes = (0,1,2), flip_probability=0.5)
        img_transforms.append(randomAffineWithRot)
        img_transforms.append(randomFlip)
        LOG.info("Applied normal training augmentation")

    Rescale = montrans.Resize((input_height, input_height, input_height))
    img_transforms.append(Rescale)
    img_transform = montrans.Compose(img_transforms)
    return img_transform


class UKBioBankDataset(Dataset):
    def __init__(self, 
                data_path: str, 
                is_training: bool, 
                out_class_num: Optional[int], 
                is_ssl,
                resize_height
                ):
        self.data_path = data_path
        self.out_class_num = out_class_num
        self.is_training = is_training
        self.resize_height = resize_height
        self.is_ssl = is_ssl

        if self.is_ssl:
            LOG.info("Applied SSL training augmentation")

        data_list = get_images_from_h5(self.data_path)

        LOG.info("DATASET: %s", self.data_path)
        LOG.info("SAMPLES: %d", len(data_list))
        self._image_data = data_list

    
    def __len__(self) -> int:
        return len(self._image_data)

    def __getitem__(self, index: int):
        scans = self._image_data[index]
        label = 100

        scans = scans[np.newaxis]
        original_height = 120
        base_transforms = [
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.CropOrPad((original_height, original_height, original_height)),
        ]
        scans = base_transforms(scans)

        self.transforms = [
                get_image_transform(self.is_training, self.is_ssl, original_height, self.resize_height)
            ]

        if self.is_ssl:
            sample1, sample2, sample = self.transforms[0](scans)
            return sample1, sample2, sample, label
        else:    
            sample = self.transforms[0](scans)
            return sample, label

class UKBioBankDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int = 128,
        num_workers: int = 12,
        num_class: Optional[int] = 2,
        is_ssl: bool = True,
        resize_height: int = 55,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_class = num_class
        self.is_ssl = is_ssl
        self.resize_height = resize_height
        
        self.data_train = '/UKB/train.h5'
        self.data_valid = '/UKB/valid.h5'

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = UKBioBankDataset(data_path = self.data_train, 
                                            out_class_num = self.num_class,
                                            is_training = True, 
                                            is_ssl = self.is_ssl,
                                            resize_height = self.resize_height)
            self.eval_data = UKBioBankDataset(data_path = self.data_valid,
                                            out_class_num = self.num_class,
                                            is_training = False, 
                                            is_ssl = self.is_ssl,
                                            resize_height = self.resize_height)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )



