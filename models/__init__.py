# PyOD models
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

MODELS = {
    "IForest": IForest,
    "KNN": KNN,
    "LOF": LOF,
    "OCSVM": OCSVM,
    "PCA": PCA,
}