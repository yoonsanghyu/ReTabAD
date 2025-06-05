from models.pyod.trainer import Trainer as PyODTrainer

TRAINERS = {
    "OCSVM": PyODTrainer,
    "LOF": PyODTrainer,
    "KNN": PyODTrainer,
    "IForest": PyODTrainer,
    "PCA": PyODTrainer,
}

MODEL_LIST = [
    "IForest",
    "KNN",
    "LOF",
    "OCSVM",
    "PCA",
]