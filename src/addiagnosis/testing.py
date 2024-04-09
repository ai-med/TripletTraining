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
from typing import List

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

LOG = logging.getLogger(__name__)


def test(config: DictConfig):
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    LOG.info("Instantiating datamodule <%s>", config.datamodule._target_)
    data: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    LOG.info("Instantiating model <%s>", config.model._target_)
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    logger: List[pl.LightningLoggerBase] = []
    if "logger" in config:
        for lg_conf in config.logger.values():
            if "_target_" in lg_conf:
                LOG.info("Instantiating logger <%s>", lg_conf._target_)
                logger.append(hydra.utils.instantiate(lg_conf))

    LOG.info("Instantiating trainer <%s>", config.trainer._target_)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    LOG.info("Starting testing!")
    trainer.test(model, data, ckpt_path=config.ckpt_path)

def cv_test(config: DictConfig, fold: int):
    # cross validation testing
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    config.datamodule.split_fold = fold
    if config.early_stop:
        ckpt_path_all = config.ckpt_path + '/checkpoints_fold_' + str(fold) + '/' +'best-val-bacc.ckpt'
    else:
        ckpt_path_all = config.ckpt_path + '/checkpoints_fold_' + str(fold) + '/' +'last.ckpt'
    LOG.info("Instantiating datamodule <%s>", config.datamodule._target_)
    data: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    LOG.info("Instantiating model <%s>", config.model._target_)
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    logger: List[pl.LightningLoggerBase] = []
    if "logger" in config:
        for lg_conf in config.logger.values():
            if "_target_" in lg_conf:
                LOG.info("Instantiating logger <%s>", lg_conf._target_)
                logger.append(hydra.utils.instantiate(lg_conf))

    LOG.info("Instantiating trainer <%s>", config.trainer._target_)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": ckpt_path_all})

    LOG.info("Starting cross-validation testing!")
    trainer.test(model, data, ckpt_path=ckpt_path_all)
