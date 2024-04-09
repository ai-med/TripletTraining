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
from PIL import Image
from sklearn.model_selection import train_test_split

LOG = logging.getLogger(__name__)

DIAGNOSIS_MAP = {"CN": 0, "AD": 1, "FTD": 2}
DIAGNOSIS_MAP_binary = {"AD": 1, "FTD": 0}


def get_image_transform(is_training: bool, resize: int = None):
    img_transforms = [
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.CropOrPad((120, 120, 120)),
    ]

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


class TaskDataset(Dataset):
    def __init__(self, 
                 path: str, 
                 data_split: list, # list of data IDs split into train, valid, test
                 is_training: bool, 
                 out_class_num: int, 
                 resize: int):
        self.path = path
        self.resize = resize
        self.is_training = is_training
        self.data_RID = data_split
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
                if group.attrs['RID'] in self.data_RID:
                    mri_data = group['MRI/T1/data'][:]
                    image_data.append(mri_data[np.newaxis])                     
                    diagnosis.append(group.attrs['DX'])
                    rid.append(group.attrs['RID'])
                else:
                    continue

        LOG.info("DATASET: %s", self.path)
        LOG.info("SAMPLES: %d", len(image_data))
        labels, counts = np.unique(diagnosis, return_counts=True)
        LOG.info("Classes: %s", pd.Series(counts, index=labels))

        self._image_data = image_data

        if self.out_class_num == 3:
            self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]
        elif self.out_class_num == 2:
            self._diagnosis = [DIAGNOSIS_MAP_binary[d] for d in diagnosis]

        self._rid = rid


    def __len__(self) -> int:
        return len(self._image_data)

    def __getitem__(self, index: int):
        label = self._diagnosis[index]
        scans = self._image_data[index]
        rids = self._rid[index]

        self.transforms = [
                get_image_transform(self.is_training, self.resize)
            ]
        assert len(scans) == len(self.transforms)

        sample = self.transforms[0](scans)

        return sample, label, rids


class TaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = None,
        data_list: str = None, # information of all data
        train_size: float = 0.8,
        batch_size: int = 32,
        num_workers: int = 4,
        num_class: int = 3,
        seed: int = 28022022,
        resize: int = 55,
        split_fold: int = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.data_list = data_list
        self.train_size = train_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_class = num_class
        self.seed = seed
        self.resize = resize
        self.split_fold = split_fold


    def setup(self, stage: Optional[str] = None):

        data_df = pd.read_csv(self.data_list)

        if self.num_class == 2:
            data_df = data_df[(data_df['DX'] == 'AD') | (data_df['DX'] == 'FTD')]
            data_df = data_df.reset_index()
        elif self.num_class == 3:
            data_df = data_df[(data_df['DX'] == 'AD') (data_df['DX'] == 'CN') | (data_df['DX'] == 'FTD')]
            data_df = data_df.reset_index()

        data_RID = data_df['RID'] if 'RID' in data_df.columns else data_df['ID']
        data_label = data_df['DX']

        assert data_RID.shape == data_label.shape

        if self.split_fold:
            self.train_RID = np.load(str(self.split_fold) + '-train.npy', allow_pickle='TRUE')
            self.valid_RID = np.load(str(self.split_fold) + '-valid.npy', allow_pickle='TRUE')
            self.test_RID = np.load(str(self.split_fold) + '-test.npy', allow_pickle='TRUE')
            
            LOG.info("Using Task data split FOLD: %s", str(self.split_fold))
            LOG.info("Using train split: %s", train_path)
        else:  
            LOG.info('no split specified, using random split')    
            self.train_RID, data_remain, self.train_label, label_remain = train_test_split(data_RID, data_label, 
                                                            train_size=self.train_size, random_state = self.seed,
                                                            stratify=data_label)
            test_size = 0.5
            self.valid_RID, self.test_RID, self.valid_label, self.test_label = train_test_split(data_remain, label_remain, 
                                                            test_size = test_size, random_state = self.seed,
                                                            stratify=label_remain)
            self.train_RID, self.valid_RID, self.test_RID = self.train_RID.values.tolist(), self.valid_RID.values.tolist(), self.test_RID.values.tolist()

        self.all_RID = data_RID.values.tolist()
        self.all_label = data_label.values.tolist()

        print('amount of all data:', len(self.all_RID))
        print('amount of training data:', len(self.train_RID))
        print('amount of validation data:', len(self.valid_RID))
        print('amount of testing data:', len(self.test_RID))
        
        
        if stage == 'fit' or stage is None:
            self.train_data = TaskDataset(self.data_path, self.train_RID, is_training=True, out_class_num = self.num_class, resize = self.resize)
            self.eval_data = TaskDataset(self.data_path, self.valid_RID, is_training=False, out_class_num = self.num_class, resize = self.resize)
        
        elif stage == 'test' and self.test_data is not None:
            self.test_data = TaskDataset(self.data_path, self.test_RID, is_training=False, out_class_num = self.num_class, resize = self.resize)


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
