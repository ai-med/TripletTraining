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
from torchmetrics import Accuracy, ConfusionMatrix, MaxMetric
from src.addiagnosis.models.utils import distLinear, DistillKL, NCELoss, Attention, HintLoss
from src.addiagnosis.modules.optimizer import LARS, CosineWarmupScheduler

LOG = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def init_weights(m: torch.Tensor):
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

def load_pretrain_model(pretrained_model, out_class_num):
    pretrained_dict = torch.load(pretrained_model)["state_dict"]
    new_pretrained_dict = dict()
    for key, value in pretrained_dict.items(): 
        if key.find('encoder.resnet') != -1:
            new_pretrained_dict[key[8:]] = value
        elif key.find('encoder') != -1:
            new_key = 'resnet.' + key[8:]
            new_pretrained_dict[new_key] = value
        elif key.find('model_student') != -1:
            new_pretrained_dict[key[14:]] = value
        elif key.find('student') != -1:
            new_pretrained_dict[key[8:]] = value
        else:
            new_pretrained_dict[key[4:]] = value

    return new_pretrained_dict


class Self_Distillation_Module(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str,
        net: torch.nn.Module,
        stu_net: torch.nn.Module,
        lr: float = 0.01,
        weight_decay: float = 1.5e-6,
        batch_size: int = 128,
        out_class_num: int = 3,
        class_loss_type: str = 'multilabel_BCE', # 'multilabel_BCE', 'CE'
        distill_loss_type: str = 'kd',
        kl_T: float = 1.0,
        weight_class_loss: float = 0.999,
        weight_div_loss: float = 0.001,
        weight_distill_loss: float = 0.0,
        is_teacher_unsupervised: bool = True,
        ema_update: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "stu_net"])
        self.batch_size = batch_size
        self.out_class_num = out_class_num
        self.class_loss_type = class_loss_type
        self.distill_loss_type = distill_loss_type
        self.kl_T = kl_T
        self.weight_class_loss = weight_class_loss
        self.weight_div_loss = weight_div_loss
        self.weight_distill_loss = weight_distill_loss
        self.is_teacher_unsupervised = is_teacher_unsupervised
        self.ema_update = ema_update

        # randomly initialize student net
        net.apply(init_weights)
        stu_net.apply(init_weights)
        self.model_student = stu_net

        if pretrained_model == None:
            net.apply(init_weights)
            self.model_teacher = net
            print('No pretrained model loaded for teacher model')
        else:
            model = net
            new_pretrained_dict = load_pretrain_model(pretrained_model, out_class_num)
            msg = model.load_state_dict(new_pretrained_dict, strict = False)
            print('======load pretrained model for teacher model successfully========')
            print("missing keys:", msg.missing_keys)
            self.model_teacher = model
            
        # freeze teacher model
        for name, param in self.model_teacher.named_parameters():               
            param.requires_grad = False
        
        for name, param in self.model_student.named_parameters():               
            param.requires_grad = True
        
        
        # loss function for classification      
        if self.class_loss_type == 'multilabel_BCE':
            self.criterion_class = torch.nn.BCEWithLogitsLoss()
          
        elif self.class_loss_type == 'CE':
            self.criterion_class = torch.nn.CrossEntropyLoss()
        
        # loss function for distillation
        self.criterion_divergence = DistillKL(self.kl_T)
        
        if self.distill_loss_type == 'kd':
            self.criterion_distill = DistillKL(self.kl_T)
        elif self.distill_loss_type == 'attention':
            self.criterion_distill = Attention()
        elif self.distill_loss_type == 'hint':   
            self.criterion_distill = HintLoss()
        else:
            raise NotImplementedError(self.distill_loss_type)
        
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_acc_teacher = Accuracy()
        self.test_acc = Accuracy()
        self.val_cmat = ConfusionMatrix(num_classes=out_class_num)
        self.test_cmat = ConfusionMatrix(num_classes=out_class_num)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_bacc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model_student(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_bacc_best.reset()

    def train_distill(self, batch: Any):
        x, y, *_ = batch
        batch_size = x.shape[0]
        
        if not self.is_teacher_unsupervised:
            logits_teacher = self.model_teacher(x, get_features = False)
        
        features_student, logits_student = self.model_student(x, get_features = None)
        features_teacher = self.model_teacher(x, get_features = True)

        if self.out_class_num == 2 and self.class_loss_type == 'multilabel_BCE':
            loss_class = self.criterion_class(logits_student.float(), y.unsqueeze(1).float())
            preds = torch.round(logits_student).squeeze(1)
        else:
            if self.class_loss_type == 'multilabel_BCE':
                one_hot_init = torch.zeros(y.shape[0], self.out_class_num)
                for i in range(y.shape[0]):
                    one_hot_init[i][y[i]] = 1
                y_one_hot = one_hot_init
                loss_class = self.criterion_class(logits_student.float().to(device), y_one_hot.float().to(device))
                preds = torch.argmax(logits_student, dim=1)
            
            elif self.class_loss_type == 'CE':                
                loss_class = self.criterion_class(logits_student, y)
                preds = torch.argmax(logits_student, dim=1)
            
            else:
                raise NotImplementedError(self.class_loss_type)
            
        if not self.is_teacher_unsupervised:
            loss_divergence = self.criterion_divergence(logits_student, logits_teacher)
        else:
            loss_divergence = self.criterion_divergence(features_student, features_teacher)

        if self.distill_loss_type == 'kd':
            loss_distill = 0

        elif self.distill_loss_type == 'attention':
            feat_s = features_student[1:-1]
            feat_t = features_teacher[1:-1]
            loss_distill = self.criterion_distill(feat_s, feat_t)
            loss_distill = sum(loss_distill)

        elif self.distill_loss_type == 'hint':   
            feat_s = features_student[-1]
            feat_t = features_teacher[-1]
            loss_distill = self.criterion_distill(feat_s, feat_t)

        else:
            raise NotImplementedError(self.distill_loss_type)
        
        loss = self.weight_class_loss * loss_class.to(device) + self.weight_div_loss * loss_divergence.to(device) + self.weight_distill_loss * loss_distill
        
        return loss.to(device), preds.to(device), y.to(device), loss_class.to(device), loss_divergence.to(device)

    def step(self, batch: Any):
        x, y, *_ = batch
        batch_size = x.shape[0]

        if not self.is_teacher_unsupervised:
            logits_teacher = self.model_teacher(x, get_features = False)
              
        features_student, logits_student = self.model_student(x, get_features = None)
        features_teacher = self.model_teacher(x, get_features = True)

        if self.out_class_num == 2 and self.class_loss_type == 'multilabel_BCE':
            loss_class = self.criterion_class(logits_student.float(), y.unsqueeze(1).float())
            preds = torch.round(logits_student).squeeze(1)
            if not self.is_teacher_unsupervised:
                preds_teacher = torch.round(logits_teacher).squeeze(1)

        else:
            if self.class_loss_type == 'multilabel_BCE':
                one_hot_init = torch.zeros(y.shape[0], self.out_class_num)
                for i in range(y.shape[0]):
                    one_hot_init[i][y[i]] = 1
                y_one_hot = one_hot_init
                loss_class = self.criterion_class(logits_student.float().to(device), y_one_hot.float().to(device))
                preds = torch.argmax(logits_student, dim=1)
                if not self.is_teacher_unsupervised:
                    preds_teacher = torch.argmax(logits_teacher, dim=1)
            
            elif self.class_loss_type == 'CE':                
                loss_class = self.criterion_class(logits_student, y)
                preds = torch.argmax(logits_student, dim=1)
                if not self.is_teacher_unsupervised:
                    preds_teacher = torch.argmax(logits_teacher, dim=1)
            
            else:
                raise NotImplementedError(self.class_loss_type)
            
        if not self.is_teacher_unsupervised:
            loss_divergence = self.criterion_divergence(logits_student, logits_teacher)
        else:
            loss_divergence = self.criterion_divergence(features_student, features_teacher)
        
        loss = self.weight_class_loss * loss_class.to(device) + self.weight_div_loss * loss_divergence.to(device)
        

        if not self.is_teacher_unsupervised:
            return loss.to(device), preds.to(device), y.to(device), preds_teacher.to(device), loss_class.to(device), loss_divergence.to(device)
        else:
            return loss.to(device), preds.to(device), y.to(device), loss_class.to(device), loss_divergence.to(device)


    def training_step(self, batch: Any, batch_idx: int):
        
        loss, preds, targets, loss_class, loss_div = self.train_distill(batch)

        # log train metrics
        student_acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_class", loss_class, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_div", loss_div, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/student_acc", student_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets, "loss_class": loss_class, "loss_div": loss_div}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # reset metrics at the end of every epoch
        self.train_acc.reset()

        ### EMA update ###
        if self.ema_update:
            self.update_teacher_model()
 
    
    def update_teacher_model(self):
        alpha=0.999
        with torch.no_grad():
            for teacher_param, student_param in zip(self.model_teacher.parameters(), self.model_student.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
    
    def validation_step(self, batch: Any, batch_idx: int): 
        if not self.is_teacher_unsupervised:  
            loss, preds, targets, preds_teacher, loss_class, loss_div = self.step(batch)
            self.val_acc_teacher.update(preds_teacher, targets)
        else:
            loss, preds, targets, loss_class, loss_div = self.step(batch)
        # log val metrics
        self.val_acc.update(preds, targets)    
        self.val_cmat.update(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss_class", loss_class, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss_div", loss_div, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets, "loss_class": loss_class, "loss_div": loss_div}

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

        if not self.is_teacher_unsupervised:  
            acc_teacher = self.val_acc_teacher.compute()
            print('ACC_teacher:', acc_teacher)

        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True)

        # compute balanced accuracy
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cmat)
        self.val_bacc_best.update(bacc)
        self.log("val/bacc", bacc, on_epoch=True, prog_bar=True)
        self.log("val/bacc_best", self.val_bacc_best.compute(), on_epoch=True)

        # reset metrics at the end of every epoch
        self.val_acc.reset()
        if not self.is_teacher_unsupervised:  
            self.val_acc_teacher.reset()
        self.val_cmat.reset()

    def test_step(self, batch: Any, batch_idx: int):
        if not self.is_teacher_unsupervised:  
            loss, preds, targets, preds_teacher, loss_class, loss_div = self.step(batch)
        else:
            loss, preds, targets, loss_class, loss_div = self.step(batch)

        # log test metrics
        self.test_acc.update(preds, targets)
        self.test_cmat.update(preds, targets)
        self.log("test/loss_class", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        acc = self.test_acc.compute()
        self.log("test/acc", acc)

        # compute balanced accuracy
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.test_cmat)
        self.log("test/bacc", bacc)

        # reset metrics at the end of every epoch
        self.test_acc.reset()
        self.test_cmat.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(
            params=self.model_student.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            batch_size=self.batch_size,
            warmup_steps=5,
            max_steps=100,
            lr=self.hparams.lr,
            end_lr=0.00001,
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train/loss"}
