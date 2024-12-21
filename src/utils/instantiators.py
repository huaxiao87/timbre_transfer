from typing import List

import hydra
from omegaconf import DictConfig
from lightning import Callback
from lightning.pytorch.loggers import Logger
from src.distillation.distillation_methods import DistillMethod

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)

from omegaconf import OmegaConf

def resolve_tuple(*args):
    return tuple(args)

OmegaConf.register_new_resolver('as_tuple', resolve_tuple)

def instantiate_distillation_methods(distillation_cfg: DictConfig) -> List[DistillMethod]:
    """Instantiates distillation methods from config."""

    distillation_methods: List[DistillMethod] = []

    if not distillation_cfg:
        log.warning("No distillation method configs found! Skipping...")
        return distillation_methods

    if not isinstance(distillation_cfg, DictConfig):
        raise TypeError("Distillation config must be a DictConfig!")

    for _, dm_conf in distillation_cfg.items():
        if isinstance(dm_conf, DictConfig) and "_target_" in dm_conf:
            log.info(f"Instantiating distillation method <{dm_conf._target_}>")
            distillation_methods.append(hydra.utils.instantiate(dm_conf, _convert_="all"))

    return distillation_methods


def instantiate_callbacks(callbacks_cfg: DictConfig, **kwargs) -> List[Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            if "ResultSaver" in cb_conf._target_:
                callbacks.append(hydra.utils.instantiate(cb_conf,
                                                         kwargs['instrument'],
                                                         kwargs['sample_rate'],
                                                         kwargs['frame_rate']))
            elif "Transpositor" in cb_conf._target_:
                callbacks.append(hydra.utils.instantiate(cb_conf,
                                                         kwargs['instrument'],
                                                         kwargs['sample_rate'],
                                                         kwargs['frame_rate']))
            else:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""

    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
