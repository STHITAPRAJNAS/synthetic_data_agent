"""Synthetic data ML models package.

Public API
----------
- ``SynthesisModel``   — abstract base class
- ``TrainingConfig``   — typed hyperparameter config
- ``TrainingResult``   — metadata returned after training
- ``CTGANTrainer``     — Conditional GAN for mixed-type tables
- ``TVAETrainer``      — Variational Autoencoder for numeric-heavy tables
- ``CopulaTrainer``    — Gaussian Copula for simple / small tables
- ``TimeGANTrainer``   — PAR model for temporal sequential data
- ``select_strategy``  — data-driven strategy selector
- ``ModelRegistry``    — quality history + artifact tracking
"""
from .base import SynthesisModel, TrainingConfig, TrainingResult
from .copula_trainer import CopulaTrainer
from .ctgan_trainer import CTGANTrainer
from .model_registry import ModelRegistry
from .strategy_selector import StrategyDecision, select_strategy
from .timegan_trainer import TimeGANTrainer
from .tvae_trainer import TVAETrainer

__all__ = [
    "SynthesisModel",
    "TrainingConfig",
    "TrainingResult",
    "CTGANTrainer",
    "TVAETrainer",
    "CopulaTrainer",
    "TimeGANTrainer",
    "select_strategy",
    "StrategyDecision",
    "ModelRegistry",
]
