from .base_trainer import BaseTrainer
from .pyod_trainer.trainer import Trainer as PyODTrainer
from .deepod_trainer.trainer import Trainer as DeepODTrainer
from .mcm_trainer.trainer import Trainer as MCMTrainer
from .drl_trainer.trainer import Trainer as DRLTrainer
from .anollm_trainer.trainer import Trainer as AnoLLMTrainer
from .zeroshotllm.trainer import Trainer as ZeroShotLLMTrainer


TRAINERS = {
    # pyod
    "OCSVM": PyODTrainer,
    "LOF": PyODTrainer,
    "KNN": PyODTrainer,
    "IForest": PyODTrainer,
    "PCA": PyODTrainer,

    # deepod
    "DeepSVDD": DeepODTrainer,
    "REPEN": DeepODTrainer,
    "RDP": DeepODTrainer,
    "RCA": DeepODTrainer,
    "GOAD": DeepODTrainer,
    "NeuTraL": DeepODTrainer,
    "SLAD": DeepODTrainer,
    "DeepIsolationForest": DeepODTrainer,

    # mcm
    "MCM": MCMTrainer,

    # drl
    "DRL": DRLTrainer,

    # anollm
    "AnoLLM": AnoLLMTrainer,

    # zeroshotllm
    "ZeroShotLLM": ZeroShotLLMTrainer,
}

MODEL_LIST = list(TRAINERS.keys())