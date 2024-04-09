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

import torch
import torch.nn as nn


def train(config: DictConfig):
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    LOG.info("Instantiating datamodule <%s>", config.datamodule._target_)
    data: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    LOG.info("Instantiating model <%s>", config.model._target_)
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for cb_conf in config.callbacks.values():
            if "_target_" in cb_conf:
                LOG.info("Instantiating callback <%s>", cb_conf._target_)
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[pl.LightningLoggerBase] = []
    if "logger" in config:
        for lg_conf in config.logger.values():
            if "_target_" in lg_conf:
                LOG.info("Instantiating logger <%s>", lg_conf._target_)
                logger.append(hydra.utils.instantiate(lg_conf))

    LOG.info("Instantiating trainer <%s>", config.trainer._target_)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    LOG.info("Starting training!")
    trainer.fit(model, data)

def cv_train(config: DictConfig, fold: int):
    # cross validation training
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    LOG.info("Instantiating datamodule <%s>", config.datamodule._target_)
    
    config.datamodule.split_fold = fold
    data: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    LOG.info("Instantiating model <%s>", config.model._target_)
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for cb_conf in config.callbacks.values():
            
            if "_target_" in cb_conf:
                if 'dirpath' in cb_conf:
                    cb_conf.dirpath = 'checkpoints_fold_' + str(fold) + '/'
                LOG.info("Instantiating callback <%s>", cb_conf._target_)
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[pl.LightningLoggerBase] = []
    if "logger" in config:
        for lg_conf in config.logger.values():
            if "_target_" in lg_conf:
                LOG.info("Instantiating logger <%s>", lg_conf._target_)
                logger.append(hydra.utils.instantiate(lg_conf))

    LOG.info("Instantiating trainer <%s>", config.trainer._target_)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    LOG.info("Starting cross-validation training!")
    trainer.fit(model, data)
