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

### Modified from https://github.com/HobbitLong/SupContrast

import logging
from typing import Any
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from src.addiagnosis.models.utils import ProjectionHead, SupConLoss
from src.addiagnosis.modules.optimizer import LARS, CosineWarmupScheduler

LOG = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def init_weights(m: torch.Tensor):
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

def load_pretrain_model(pretrained_model):

    pretrained_dict = torch.load(pretrained_model)["state_dict"]

    new_pretrained_dict = dict()
    for key, value in pretrained_dict.items():
        if key.find('fc') != -1:
            continue
        
        if key.find('encoder.resnet') != -1:
            new_pretrained_dict[key[15:]] = value
        elif key.find('encoder') != -1:
            new_pretrained_dict[key[8:]] = value
        elif key.find('model_student') != -1:
            new_pretrained_dict[key[14:]] = value
            
        elif key.find('student') != -1:
            new_pretrained_dict[key[8:]] = value
        
        else:
            new_pretrained_dict[key[4:]] = value

    return new_pretrained_dict


class Sup_Con_Learning_Module(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str,
        net: torch.nn.Module,
        lr: float = 0.01,
        weight_decay: float = 0.0005,
        batch_size: int = 256,
        encoder_out_dim: int = 512,
        z_dim: int = 128,
        T: float = 0.1, 
        method: str = 'SupCon',
        end_lr: float = 0.0001,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.method = method
        self.end_lr = end_lr
        
        # initialize encoder
        net.apply(init_weights)
        self.encoder = net

        if pretrained_model != None:
            model = net
            new_pretrained_dict = load_pretrain_model(pretrained_model)
            msg = model.load_state_dict(new_pretrained_dict, strict = False)
            self.encoder = model            
            print('======load pretrained encoder successfully========')
            print("missing keys:", msg.missing_keys)
        
        self.encoder.to(device)

        # get project head
        self.projection_head = ProjectionHead(num_layer = 2,
                                              input_dim = encoder_out_dim, 
                                              hidden_dim = encoder_out_dim, 
                                              output_dim = z_dim,
                                              last_bn = False)
   
        # loss function 
        self.SupConLoss = SupConLoss(temperature= T, base_temperature=0.07)
        

    def forward(self, x: torch.Tensor):
        # encoder is only a backbone
        return self.encoder(x)
    
    def shared_step(self, batch):
        x1, x2, x, y = batch
        images = torch.cat([x1, x2], dim=0)
        batch_size = y.shape[0]
        
        features = F.normalize(self.projection_head(self.forward(images)), dim = 1)
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if self.method == 'SupCon':
            loss = self.SupConLoss(features, y)
        elif self.method == 'SimCLR':
            loss = self.SupConLoss(features)

        return loss

    def training_step(self, batch: Any, batch_idx: int):       
        loss = self.shared_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}
 
    def validation_step(self, batch: Any, batch_idx: int):       
        loss = self.shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.shared_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = LARS(
            self.parameters(),
            lr=0,  # Initialize with a LR of 0
            weight_decay=self.hparams.weight_decay,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm
        )

        lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            batch_size=self.batch_size,
            warmup_steps=10,
            max_steps=100,
            lr=self.hparams.lr,
            end_lr = self.end_lr,
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train/loss"}

def exclude_bias_and_norm(p):
    return p.ndim == 1