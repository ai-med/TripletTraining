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
from typing import Any, List

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, ConfusionMatrix, MaxMetric, F1Score
from src.addiagnosis.models.utils import distLinear
from src.addiagnosis.modules.optimizer import CosineWarmupScheduler

LOG = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def init_weights(m: torch.Tensor):
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

def load_pretrain_model(pretrained_model, num_class):

    pretrained_dict = torch.load(pretrained_model)["state_dict"]

    new_pretrained_dict = dict()
    for key, value in pretrained_dict.items():
        if key.find('model_student') != -1:
            new_pretrained_dict[key[14:]] = value
        
        elif key.find('encoder.resnet') != -1:
            new_pretrained_dict[key[8:]] = value
        
        elif key.find('encoder') != -1:
            new_key = 'resnet.' + key[8:]
            new_pretrained_dict[new_key] = value
        
        elif key.find('student') != -1:
            new_pretrained_dict[key[8:]] = value
                 
        else:
            new_pretrained_dict[key[4:]] = value

    return new_pretrained_dict


class Transfer_Learning_DiagnosisModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        out_class_num: int = 3,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        if pretrained_model == None:
            net.apply(init_weights)
            self.net = net
            print('No pretrained model loaded')
        else:
            model = net
            msg = model.load_state_dict(load_pretrain_model(pretrained_model, out_class_num), strict = False)
            print('======load pretrained model successfully========')
            print("missing keys:", msg.missing_keys)

            self.net = model
   
        self.out_class_num = out_class_num

        # loss function
        if self.out_class_num == 2:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.out_class_num == 3:
            self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy().to(device)
        self.val_acc = Accuracy().to(device)
        self.test_acc = Accuracy().to(device)
        self.val_cmat = ConfusionMatrix(num_classes=out_class_num).to(device)
        self.test_cmat = ConfusionMatrix(num_classes=out_class_num).to(device)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_bacc_best = MaxMetric()

        self.test_f1_micro = F1Score(task="multiclass", num_classes=3, average='micro').to(device)
        self.test_f1_macro = F1Score(task="multiclass", num_classes=3, average='macro').to(device)


    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_bacc_best.reset()

    def step(self, batch: Any):
        x, y, *_ = batch
        logits = self.forward(x)

        features = None 

        if self.out_class_num == 2:
            loss = self.criterion(logits.float(), y.unsqueeze(1).float())
            preds = torch.round(logits).squeeze(1)

        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

        return loss.to(device), preds.to(device), y.to(device), features 

    def training_step(self, batch: Any, batch_idx: int):
        
        loss, preds, targets, _ = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # reset metrics at the end of every epoch
        self.train_acc.reset()
 
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, features = self.step(batch)
        # log val metrics
        self.val_acc.update(preds.to(device), targets.to(device))
        self.val_cmat.update(preds.to(device), targets.to(device))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def _get_balanced_accuracy_from_confusion_matrix(self, confusion_matrix: ConfusionMatrix):
        # Confusion matrix whose i-th row and j-th column entry indicates
        # the number of samples with true label being i-th class and
        # predicted label being j-th class.
        cmat = confusion_matrix.compute()
        LOG.info("Confusion matrix:\n%s", cmat)

        return (cmat.diag() / cmat.sum(dim=1)).mean()

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.log("val/acc", acc, on_epoch=True)

        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True)

        # compute balanced accuracy
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cmat)
        self.val_bacc_best.update(bacc)
        self.log("val/bacc", bacc, on_epoch=True, prog_bar=True)
        self.log("val/bacc_best", self.val_bacc_best.compute(), on_epoch=True)

        # reset metrics at the end of every epoch
        self.val_acc.reset()
        self.val_cmat.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        # log test metrics
        self.test_acc.update(preds, targets)
        self.test_f1_micro.update(preds, targets)
        self.test_f1_macro.update(preds, targets)
        self.test_cmat.update(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        acc = self.test_acc.compute()
        self.log("test/acc", acc)

        f1_micro = self.test_f1_micro.compute()
        self.log("test/f1_micro", f1_micro)
        f1_macro = self.test_f1_macro.compute()
        self.log("test/f1_macro", f1_macro)

        # compute balanced accuracy
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.test_cmat)
        self.log("test/bacc", bacc)

        
        LOG.info("test/acc: %.6f", acc)
        LOG.info("test/bacc: %.6f", bacc)
        LOG.info("test/f1_micro: %.6f", f1_micro)
        LOG.info("test/f1_macro: %.6f", f1_macro)

        # reset metrics at the end of every epoch
        self.test_acc.reset()
        self.test_f1_micro.reset()
        self.test_f1_macro.reset()
        self.test_cmat.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer = torch.optim.AdamW(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
      
        lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            batch_size=64,
            warmup_steps=3,
            max_steps=50,
            lr=self.hparams.lr,
            end_lr=0.00001,
        )


        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train/loss"}
