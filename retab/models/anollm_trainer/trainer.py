from addict import Dict
import torch

from retab.datasets import Preprocessor
from retab.models import BaseTrainer
from retab.utils import get_summary_metrics

from retab.models.anollm_trainer.anollm import AnoLLM

class Trainer(BaseTrainer):
    def __init__(self, data_params: Dict, model_params: Dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)

        # Will be in model_params
        self.llm = 'HuggingFaceTB/SmolLM-135M'
        self.max_length_dict = {col: 1000 for col in self.column_names}
        self.model = AnoLLM(llm=self.llm, max_length_dict=self.max_length_dict)


    def train(self):
        self.model.fit(data=self.X_train, 
                       column_names=self.column_names)

    @torch.no_grad()
    def evaluate(self): 
        scores = self.model.decision_function(
            data=self.X_train, 
            column_names=self.column_names
        )
        metrics = get_summary_metrics(y_true=self.y_test, y_pred=scores.mean(axis=1))
        return metrics