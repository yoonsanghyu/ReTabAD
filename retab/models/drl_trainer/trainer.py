import os
import torch
from addict import Dict

from .drl import DRL
from retab.utils import get_summary_metrics
from retab.datasets import Preprocessor
from retab.models import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, data_params: dict, model_params: dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)
        
        self.device = meta_info.device
        model_params.data_dim = self.preprocessor.nfeatures
        self.model = DRL(model_params, self.device)

        self.ckpt_path = os.path.join(meta_info.checkpoint_path, meta_info.data_name, meta_info.model_name, meta_info.exp_id, f'{meta_info.seed}.pth')
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

    def train(self):
        self.model.fit(self.trainloader)
        self.save()

    @torch.no_grad()
    def evaluate(self):
        self.load()
        scores = self.model.decision_function(self.testloader)
        metrics = get_summary_metrics(y_true=self.y_test, y_pred=scores)
        return metrics
    
    def save(self):
        torch.save(self.model, self.ckpt_path)

    def load(self):
        self.model = torch.load(self.ckpt_path)