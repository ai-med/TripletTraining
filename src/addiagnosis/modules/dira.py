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

### Modified from https://github.com/fhaghighi/DiRA

import logging
from typing import Any, List, Optional
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchio as tio
from src.addiagnosis.models.utils import BarlowTwinsLoss, ProjectionHead, ClassificationHead, SupConLoss, VICRegLoss
from src.addiagnosis.modules.optimizer import LARS, CosineWarmupScheduler

LOG = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_embeddings = []

@torch.no_grad()
def init_weights(m: torch.Tensor):
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

@torch.no_grad()
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def load_pretrain_model(pretrained_model):
    pretrained_dict = torch.load(pretrained_model)["state_dict"]
    new_pretrained_dict = dict()
    for key, value in pretrained_dict.items():
        if key.find('encoder.resnet') != -1:
            new_pretrained_dict[key[15:]] = value
        else:  
            new_pretrained_dict[key[4:]] = value
    return new_pretrained_dict

def load_pretrain_dir_model(pretrained_model):
    pretrained_dict = torch.load(pretrained_model)["state_dict"]
    pretrained_encoder_dict = dict()
    pretrained_decoder_dict = dict()
    for key, value in pretrained_dict.items():
        if key.find('encoder') != -1:
            pretrained_encoder_dict[key[8:]] = value
        elif key.find('decoder') != -1:
            pretrained_decoder_dict[key[8:]] = value
        else:  
            pretrained_encoder_dict[key[4:]] = value
            pretrained_decoder_dict[key[4:]] = value
    return pretrained_encoder_dict, pretrained_decoder_dict


class Dira_Module(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str,
        pretrained_dir_model: str,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        discriminator: torch.nn.Module,
        dis_mode: str = 'barlow',
        lr: float = 0.0001,
        lr_d: float = 0.001,
        weight_decay: float = 0.0005,
        batch_size: int = 256,
        encoder_out_dim: int = 512,
        z_dim: int = 128,
        lambda_coeff: float = 5e-3,
        warmup_epochs: int = 10,
        lamda_dis: float = 100.0,
        lamda_res: float = 1.0,
        lamda_adv: float = 1.0,
        lamda_class: Optional[float] = 1.0,
        adv_loss_type: str = 'BCE', # BCE or MSE for adversarial loss
        opt_g: str = 'adamw', # adamw or lars
        opt_d: str = 'adamw', # adamw or lars
        sche_g: str = 'cosine', # ReduceLROnPlateau or cosine
        sche_d: str = 'cosine', # ReduceLROnPlateau or cosine
        train_modules: str = 'dira', # choose to train the whole 'dira' or only the 'dir' module
        with_classification: str = None, # None, 'diagnosis' or other classification task
        T: Optional[float] = 0.1,
        lambda_in: Optional[float] = 25.0,
        lambda_va: Optional[float] = 25.0,
        lambda_co: Optional[float] = 1.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["encoder", "decoder", "discriminator"])
        self.z_dim = z_dim
        self.lambda_coeff = lambda_coeff
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size

        self.dis_mode = dis_mode
        self.with_classification = with_classification

        self.lamda_dis = lamda_dis
        self.lamda_res = lamda_res
        self.lamda_adv = lamda_adv
        self.lamda_class = lamda_class
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.sche_g = sche_g
        self.sche_d = sche_d
        self.train_modules = train_modules
        self.adv_loss_type = adv_loss_type

        # initialize nets
        encoder.apply(init_weights)
        self.encoder = encoder
        decoder.apply(init_weights)
        self.decoder = decoder

        if pretrained_model != None:
            model = encoder
            new_pretrained_dict = load_pretrain_model(pretrained_model)
            msg = model.load_state_dict(new_pretrained_dict, strict = False)
            print('======load pretrained model successfully========')
            print("missing keys:", msg.missing_keys)
            self.encoder = model
            print('load pretrained model for encoder successfully')
        elif pretrained_dir_model != None:
            pretrained_encoder_dict, pretrained_decoder_dict = load_pretrain_dir_model(pretrained_dir_model)
            msg_en = self.encoder.load_state_dict(pretrained_encoder_dict, strict = False)
            print('======load pretrained encoder model successfully========')
            print("missing keys:", msg_en.missing_keys)
            msg_de = self.decoder.load_state_dict(pretrained_decoder_dict, strict = False)
            print('======load pretrained decoder model successfully========')
            print("missing keys:", msg_de.missing_keys)
        
        
        discriminator.apply(weights_init_normal)
        self.discriminator = discriminator
        
        self.encoder.to(device)
        self.decoder.to(device)
        self.discriminator.to(device)   
        
        if self.with_classification != None:
            class_dim = 2 if self.with_classification == 'diagnosis' else 3
            self.classification_head = ClassificationHead(input_dim=encoder_out_dim, 
                                                          hidden_dim=encoder_out_dim, 
                                                          output_dim=class_dim)
            self.classifiation_loss = nn.CrossEntropyLoss()
            self.classifiation_loss_L2 = nn.MSELoss(reduction='mean')
   
        # loss function 
        if self.dis_mode == 'barlow':
            # get project head
            self.projection_head = ProjectionHead(
                                    num_layer = 2,
                                    input_dim=encoder_out_dim, 
                                    hidden_dim=encoder_out_dim * 2, 
                                    output_dim=z_dim,
                                    last_bn=True)
            self.barlow_loss = BarlowTwinsLoss(batch_size=batch_size, 
                                           lambda_coeff=lambda_coeff, 
                                           z_dim=z_dim)
        elif self.dis_mode == 'SupCon' or self.dis_mode == 'SimCLR':
            # get project head
            self.projection_head = ProjectionHead(
                                    num_layer = 2,
                                    input_dim=encoder_out_dim, 
                                    hidden_dim=encoder_out_dim, 
                                    output_dim=z_dim,
                                    last_bn=False)
            self.SupConLoss = SupConLoss(temperature= T, base_temperature=0.07)

        elif self.dis_mode == 'vicreg':
            # get project head
            self.projection_head = ProjectionHead(
                                    num_layer = 2,
                                    input_dim=encoder_out_dim, 
                                    hidden_dim=encoder_out_dim * 2, 
                                    output_dim=z_dim,
                                    last_bn=False)
            self.vicreg_loss = VICRegLoss(batch_size=batch_size, 
                                    lambda_in=lambda_in,
                                    lambda_va = lambda_va,
                                    lambda_co = lambda_co,
                                    z_dim=z_dim)
        elif self.dis_mode == 'CE':
            # get project head
            self.projection_head = ProjectionHead(
                                    num_layer = 2,
                                    input_dim=encoder_out_dim, 
                                    hidden_dim=encoder_out_dim * 2, 
                                    output_dim=z_dim,
                                    last_bn=False)
            self.class_loss = torch.nn.CrossEntropyLoss()
        
        else:
            raise ValueError(self.dis_mode, '--> this discriminative mode has not been implemented')

        self.restore_loss = nn.MSELoss(reduction='mean')
        if adv_loss_type == 'BCE':
            self.adversarial_loss = nn.BCEWithLogitsLoss(reduction='mean')
        elif adv_loss_type == 'MSE':
            self.adversarial_loss = nn.MSELoss(reduction='mean')
        

    def forward(self, x: torch.Tensor):
        out, features = self.encoder(x, get_all_features = True)
        restored_x = self.decoder(features)
        return out, features, restored_x
    
    def generator_step(self, batch):
        x1, x2, x, y = batch
        out1, features1, restored_x1 = self.forward(x1)
        out2, features2, restored_x2 = self.forward(x2)

        ##### Discrimination part ##### 
        if self.dis_mode == 'barlow': 
            z1 = self.projection_head(out1)
            z2 = self.projection_head(out2)

            dis_loss = self.barlow_loss(z1, z2)

            if self.with_classification != None:
                logits_x1 = self.classification_head(out1)
                class_loss = self.classifiation_loss(logits_x1, y)

                logits_restored_x1 = self.classification_head(self.encoder(restored_x1))
                class_loss += self.classifiation_loss_L2(logits_restored_x1, logits_x1)
        
        elif self.dis_mode == 'SupCon' or self.dis_mode == 'SimCLR':
            images = torch.cat([x1, x2], dim=0)
            batch_size = y.shape[0]
            
            features = F.normalize(self.projection_head(self.encoder(images)), dim = 1)
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if self.dis_mode == 'SupCon':
                loss = self.SupConLoss(features, y)
            elif self.dis_mode == 'SimCLR':
                loss = self.SupConLoss(features)
            dis_loss = loss
        
        elif self.dis_mode == 'vicreg':
            z1 = self.projection_head(out1)
            z2 = self.projection_head(out2)

            dis_loss, _, _, _ = self.vicreg_loss(z1, z2)
            
            if self.with_classification != None:
                logits_x1 = self.classification_head(out1)
                class_loss = self.classifiation_loss(logits_x1, y)

                logits_restored_x1 = self.classification_head(self.encoder(restored_x1))
                class_loss += self.classifiation_loss_L2(logits_restored_x1, logits_x1)

        ##### Restoration part #####
        valid = torch.ones(x.size(0), 1)
        valid = valid.type_as(x)
        fake = torch.zeros(restored_x1.size(0), 1)
        fake = fake.type_as(restored_x1)

        res_loss = self.restore_loss(x, restored_x1)

        if self.train_modules == 'dira':
            if self.adv_loss_type == 'wgan':
                g_adv_loss = -torch.mean(self.discriminator(restored_x1))
            else:
                g_adv_loss = self.adversarial_loss(self.discriminator(restored_x1), valid)
            # g_loss = dis_loss * self.lamda_dis + res_loss * self.lamda_res + g_adv_loss * self.lamda_adv
            g_loss = dis_loss * self.lamda_dis + (res_loss + g_adv_loss) * self.lamda_res
            if self.with_classification != None:
                g_loss += class_loss * self.lamda_class
                return g_loss.to(device), dis_loss.to(device), res_loss.to(device), g_adv_loss.to(device), class_loss.to(device)
            else:
                return g_loss.to(device), dis_loss.to(device), res_loss.to(device), g_adv_loss.to(device)
        else:
            g_loss = dis_loss * self.lamda_dis + res_loss * self.lamda_res
            if self.with_classification != None:
                g_loss += class_loss * self.lamda_class
                return g_loss.to(device), dis_loss.to(device), res_loss.to(device), class_loss.to(device)
            else:
                return g_loss.to(device), dis_loss.to(device), res_loss.to(device)
    
    def discriminator_step(self, batch):
        x1, x2, x, y = batch
        out1, features1, restored_x1 = self(x1)

        ##### Adversarial part #####

        valid = torch.ones(x.size(0), 1)
        valid = valid.type_as(x)
       
        if self.adv_loss_type == 'wgan':
            adv_loss = -torch.mean(self.discriminator(x)) + torch.mean(self.discriminator(restored_x1.detach()))
        else:
            real_loss = self.adversarial_loss(self.discriminator(x), valid)

            fake = torch.zeros(restored_x1.size(0), 1)
            fake = fake.type_as(restored_x1)
            fake_loss = self.adversarial_loss(self.discriminator(restored_x1.detach()), fake)
            adv_loss = (real_loss + fake_loss) / 2


        ##### Total Loss #####
        d_loss =  adv_loss * self.lamda_adv

        return d_loss.to(device), adv_loss.to(device)

    
    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int = 0):
        if self.train_modules == 'dira':
            if optimizer_idx == 0:
                if self.with_classification != None:
                    g_loss, dis_loss, res_loss, g_adv_loss, class_loss = self.generator_step(batch)
                else:
                    g_loss, dis_loss, res_loss, g_adv_loss = self.generator_step(batch)    
                self.log("train/g_loss", g_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("train/dis_loss", dis_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("train/res_loss", res_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("train/g_adv_loss", g_adv_loss, on_step=False, on_epoch=True, prog_bar=False)

                if self.with_classification != None:
                    self.log("train/class_loss", class_loss, on_step=False, on_epoch=True, prog_bar=False)
                    return {"loss": g_loss, "dis_loss": dis_loss, "res_loss": res_loss, "g_adv_loss": g_adv_loss, "class_loss": class_loss}

                return {"loss": g_loss, "dis_loss": dis_loss, "res_loss": res_loss, "g_adv_loss": g_adv_loss}
            elif optimizer_idx == 1:
                d_loss, adv_loss = self.discriminator_step(batch)
                clip_value = 0.1
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)   
                self.log("train/d_loss", d_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("train/adv_loss", adv_loss, on_step=False, on_epoch=True, prog_bar=False)

                return {"loss": d_loss, "adv_loss": adv_loss}
        else:
            if self.with_classification != None:
                g_loss, dis_loss, res_loss, class_loss = self.generator_step(batch)
            else:
                g_loss, dis_loss, res_loss = self.generator_step(batch)    
            self.log("train/g_loss", g_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/dis_loss", dis_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/res_loss", res_loss, on_step=False, on_epoch=True, prog_bar=False)
            if self.with_classification != None:
                self.log("train/class_loss", class_loss, on_step=False, on_epoch=True, prog_bar=False)
                return {"loss": g_loss, "dis_loss": dis_loss, "res_loss": res_loss, "class_loss": class_loss}

            return {"loss": g_loss, "dis_loss": dis_loss, "res_loss": res_loss}
 
    def validation_step(self, batch: Any, batch_idx: int):
        self.validation_batch = batch
        if self.train_modules == 'dira':
            if self.with_classification != None:
                g_loss, dis_loss, res_loss, g_adv_loss, class_loss = self.generator_step(batch)
                self.log("val/class_loss", class_loss, on_step=False, on_epoch=True, prog_bar=False)
            else:
                g_loss, dis_loss, res_loss, g_adv_loss = self.generator_step(batch)

            d_loss, adv_loss = self.discriminator_step(batch)  
            total_loss = g_loss + d_loss
            self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/dis_loss", dis_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/res_loss", res_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/g_adv_loss", g_adv_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/d_loss", d_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/adv_loss", adv_loss, on_step=False, on_epoch=True, prog_bar=False)
         
            return {"loss": total_loss, "dis_loss": dis_loss, "res_loss": res_loss, "g_adv_loss": g_adv_loss,
                    "d_loss": d_loss, "adv_loss": adv_loss}
 
        else:
            if self.with_classification != None:
                g_loss, dis_loss, res_loss, class_loss = self.generator_step(batch)
                self.log("val/class_loss", class_loss, on_step=False, on_epoch=True, prog_bar=False)
            else:
                g_loss, dis_loss, res_loss = self.generator_step(batch)    
            self.log("val/loss", g_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/dis_loss", dis_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/res_loss", res_loss, on_step=False, on_epoch=True, prog_bar=False)

            return {"loss": g_loss, "dis_loss": dis_loss, "res_loss": res_loss}

    def on_validation_epoch_end(self):
        x1, _, _, _ = self.validation_batch
        _, _, restored_x1 = self(x1)

        # log sampled images
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        restored_x1 = rescale(restored_x1[0].cpu())[:, :, 23, :]
        restored_x1 = restored_x1.squeeze(dim = 2)
        grid = torchvision.utils.make_grid(restored_x1)
        self.logger.experiment.add_image("restored_images", grid, self.current_epoch)
    
    def test_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.generator_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss}

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if self.train_modules == 'dira':
            # update generator every step
            if optimizer_idx == 0:
                optimizer.step(closure=optimizer_closure)

            # update discriminator every 3 steps
            elif optimizer_idx == 1:
                if (batch_idx + 1) % 3 == 0:
                    # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                    optimizer.step(closure=optimizer_closure)
                else:
                    # call the closure by itself to run `training_step` + `backward` without an optimizer step
                    optimizer_closure()
        else:
            optimizer.step(closure=optimizer_closure)
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        params_ed = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        def opt(opt_name, params, lr, weight_decay):
            if opt_name == 'adamw':
                opt = torch.optim.AdamW(params = params, lr=lr, weight_decay=weight_decay)
            elif opt_name == 'lars':
                opt = LARS(params, lr=0, weight_decay=weight_decay,
                           weight_decay_filter=exclude_bias_and_norm,
                           lars_adaptation_filter=exclude_bias_and_norm)
            else:
                raise ValueError('Have not implemented ' + opt_name)
            return opt
        
        def sche(sche_name, optimizer, lr, batch_size, freq = 1):
            if sche_name == 'ReduceLROnPlateau':
                sche = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, cooldown=5, min_lr=1e-6, verbose=True)
            elif sche_name == 'cosine':
                sche = CosineWarmupScheduler(
                            optimizer=optimizer,
                            batch_size=batch_size,
                            warmup_steps=10,
                            max_steps=100,
                            lr=lr,
                            freq=freq,
                        )
            else:
                raise ValueError('Have not implemented ' + sche_name)
            return sche

        optimizer_g = opt(self.opt_g, params_ed, self.hparams.lr, self.hparams.weight_decay)
        optimizer_d = opt(self.opt_d, self.discriminator.parameters(), 
                        lr=self.hparams.lr_d, weight_decay=self.hparams.weight_decay)

        lr_scheduler_g = sche(self.sche_g, optimizer_g, self.hparams.lr, self.batch_size, freq = 1)
        lr_scheduler_d = sche(self.sche_d, optimizer_d, self.hparams.lr_d, self.batch_size, freq = 3)

        scheduler_g = {
            'scheduler': lr_scheduler_g,
            'interval': 'epoch',
            "monitor": "train/g_loss",
            'frequency': 1,
        }

        scheduler_d = {
            'scheduler': lr_scheduler_d,
            'interval': 'epoch',
            "monitor": "train/d_loss",
            'frequency': 3,
        }

        if self.train_modules == 'dir':
            return [optimizer_g], [scheduler_g]
        elif self.train_modules == 'dira':
            return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

def exclude_bias_and_norm(p):
    return p.ndim == 1