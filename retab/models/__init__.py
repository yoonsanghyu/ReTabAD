from .base_trainer import BaseTrainer
from .pyod_trainer.trainer import Trainer as PyODTrainer
from .anollm_trainer.trainer import Trainer as AnoLLMTrainer

TRAINERS = {
    "OCSVM": PyODTrainer,
    "LOF": PyODTrainer,
    "KNN": PyODTrainer,
    "IForest": PyODTrainer,
    "PCA": PyODTrainer,
    "AnoLLM": AnoLLMTrainer
}

MODEL_LIST = [
    "IForest",
    "KNN",
    "LOF",
    "OCSVM",
    "PCA",
]