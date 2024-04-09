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

import logging
from typing import Optional
import h5py
import monai.transforms as montrans
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torchio as tio
from src.addiagnosis.datamodules.data_transform import *

LOG = logging.getLogger(__name__)

DIAGNOSIS_MAP_with_FTD = {"CN": 0, "Dementia": 1, "FTD": 2}
DIAGNOSIS_MAP_with_FTD_with_MCI = {"CN": 0, "MCI": 1, "Dementia": 2, "FTD": 3}
DIAGNOSIS_MAP_binary_dementia = {"Dementia": 0, "FTD": 1}

def get_image_transform(is_training: bool, 
                        is_ssl: bool, 
                        original_height: int = 120,
                        resize: int = None):
 
    if is_ssl:
        print("Applied SSL training augmentation")
        return BarlowTwinsTransform(is_training = is_training,
                                    original_height = original_height, 
                                    input_height = resize)
    
    img_transforms = []
    if is_training:
        randomAffineWithRot = tio.RandomAffine(
            scales=0.05,
            degrees=8,  # +-8 degree in each dimension
            translation=8,  # +-8 pixels offset in each dimension.
            image_interpolation="linear",
            default_pad_value="otsu",
            p=0.5,
        )
        img_transforms.append(randomAffineWithRot)
    
    if resize:
        Rescale = montrans.Resize((resize, resize, resize)) 
        img_transforms.append(Rescale)

    img_transform = montrans.Compose(img_transforms)
    return img_transform



class ADNI_NIFDDataset(Dataset):
    def __init__(self, path: str, is_ssl: bool, is_training: bool, 
                 out_class_num: int, resize: int):
        self.path = path
        self.is_ssl = is_ssl
        self.is_training = is_training
        self.resize = resize
        
        self.out_class_num = out_class_num
        self._load()

    def _load(self):
        image_data = []
        diagnosis = []
        rid = []
        
        with h5py.File(self.path, mode='r') as file:
            for name, group in file.items():
                if name == "stats":
                    continue
                mri_data = group['MRI/T1/data'][:]
                mri_data = mri_data[np.newaxis]
                image_data.append(mri_data)
                diagnosis.append(group.attrs['DX'])
                rid.append(group.attrs['UID'])

        LOG.info("DATASETS: %s", self.path)
        LOG.info("SAMPLES: %d", len(image_data))
        labels, counts = np.unique(diagnosis, return_counts=True)
        LOG.info("Classes: %s", pd.Series(counts, index=labels))

        self._image_data = image_data

        if self.out_class_num == 3:
            self._diagnosis = [DIAGNOSIS_MAP_with_FTD[d] for d in diagnosis]
        elif self.out_class_num == 4:
            self._diagnosis = [DIAGNOSIS_MAP_with_FTD_with_MCI[d] for d in diagnosis]
        elif self.out_class_num == 2:
            self._diagnosis = [DIAGNOSIS_MAP_binary_dementia[d] for d in diagnosis]

        self._rid = rid


    def __len__(self) -> int:
        return len(self._image_data)

    def __getitem__(self, index: int):
        label = self._diagnosis[index]
        scans = self._image_data[index]

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


class ADNI_NIFDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = None,
        data_list: str = None,
        batch_size: int = 64,
        num_workers: int = 12,
        num_class: int = 3,
        seed: int = 0,
        resize: int = None,  
        is_ssl: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_class = num_class
        self.seed = seed
        self.resize = resize
        self.is_ssl = is_ssl

        self.train_data_path = 'adni_nifd_train.h5'
        self.valid_data_path = 'adni_nifd_valid.h5'
        self.test_data_path = 'adni_nifd_test.h5'

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit' or stage is None:
            self.train_data = ADNI_NIFDDataset(self.train_data_path, is_training=True, is_ssl = self.is_ssl, out_class_num = self.num_class, resize = self.resize)
            self.eval_data = ADNI_NIFDDataset(self.valid_data_path, is_training=False, is_ssl = self.is_ssl, out_class_num = self.num_class, resize = self.resize)
        
        elif stage == 'test' and self.test_data is not None:
            self.test_data = ADNI_NIFDDataset(self.test_data_path, is_training=False, is_ssl = self.is_ssl, out_class_num = self.num_class, resize = self.resize)

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
            self.eval_data, batch_size=self.batch_size, num_workers=self.num_workers // 2, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

