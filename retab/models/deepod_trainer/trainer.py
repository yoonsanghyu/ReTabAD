import torch
from addict import Dict
from deepod.models.tabular import DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, SLAD, DeepIsolationForest

from retab.utils import get_summary_metrics
from retab.datasets import Preprocessor
from retab.models import BaseTrainer


DeepOD_MODELS = {
    "DeepSVDD": DeepSVDD,
    "REPEN": REPEN,
    "RDP": RDP,
    "RCA": RCA,
    "GOAD": GOAD,
    "NeuTraL": NeuTraL,
    "SLAD": SLAD,
    "DeepIsolationForest": DeepIsolationForest,
}


class Trainer(BaseTrainer):
    def __init__(self, data_params: Dict, model_params: Dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)

        self.model_name = meta_info.model_name

        model_obj = DeepOD_MODELS[self.model_name]
        if hasattr(model_obj, "random_state"):
            model_params.random_state = meta_info.seed
            self.model = model_obj(**model_params)
        else:
            self.model = model_obj(**model_params)

        self.X_train_cont = torch.tensor(self.X_train_cont, dtype=torch.float32)
        self.X_test_cont = torch.tensor(self.X_test_cont, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

    def train(self):
        self.model.fit(X=self.X_train_cont)

    @torch.no_grad()
    def evaluate(self):        
        ascs = self.model.decision_function(self.X_test_cont)
        metrics = get_summary_metrics(y_true=self.y_test.cpu().numpy(), y_pred=ascs)
        return metrics