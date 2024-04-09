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

### Modified from https://github.com/facebookresearch/vicreg

import logging
from typing import Any
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from src.addiagnosis.models.utils import VICRegLoss, ProjectionHead
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
        new_pretrained_dict[key[4:]] = value
    return new_pretrained_dict


class VICReg_Module(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str,
        net: torch.nn.Module,
        lr: float = 0.01,
        end_lr: float = 0.002,
        weight_decay: float = 0.0005,
        batch_size: int = 256,
        encoder_out_dim: int = 512,
        z_dim: int = 1024,
        lambda_in: float = 25.0,
        lambda_va: float = 25.0,
        lambda_co: float = 1.0,
        warmup_epochs: int = 10,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.z_dim = z_dim
        self.lambda_in = lambda_in
        self.lambda_va = lambda_va
        self.lambda_co = lambda_co
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.end_lr = end_lr
        
        # initialize encoder
        net.apply(init_weights)
        self.encoder = net

        if pretrained_model != None:
            model = net
            new_pretrained_dict = load_pretrain_model(pretrained_model)
            model.load_state_dict(new_pretrained_dict, strict = False)
            self.encoder = model            
            print('load pretrained model for encoder successfully')
        
        self.encoder.to(device)

        # get project head
        self.projection_head = ProjectionHead(
                                    num_layer = 3,
                                    input_dim=encoder_out_dim, 
                                    hidden_dim=encoder_out_dim * 4, 
                                    output_dim=z_dim,
                                    last_bn=False)
   
        # loss function 
        self.vicreg_loss = VICRegLoss(batch_size=batch_size, 
                                      lambda_in=lambda_in,
                                      lambda_va = lambda_va,
                                      lambda_co = lambda_co,
                                      z_dim=z_dim)
        

    def forward(self, x: torch.Tensor):
        return self.encoder(x, get_features = True)
    
    def shared_step(self, batch):
        x1, x2, _, y = batch

        z1 = self.projection_head(self.forward(x1))
        z2 = self.projection_head(self.forward(x2))

        return self.vicreg_loss(z1, z2)

    def training_step(self, batch: Any, batch_idx: int):       
        loss, repr_loss, std_loss, cov_loss = self.shared_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/repr_loss", repr_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/std_loss", std_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/cov_loss", cov_loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}
 
    def validation_step(self, batch: Any, batch_idx: int):       
        loss, repr_loss, std_loss, cov_loss = self.shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/repr_loss", repr_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/std_loss", std_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/cov_loss", cov_loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int):
        loss, repr_loss, std_loss, cov_loss = self.shared_step(batch)
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