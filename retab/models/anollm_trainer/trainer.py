from addict import Dict
import torch
import torch.distributed as dist

from retab.datasets import Preprocessor
from retab.models import BaseTrainer
from retab.utils import get_summary_metrics

from retab.models.anollm_trainer.anollm import AnoLLM

class Trainer(BaseTrainer):
    def __init__(self, data_params: Dict, model_params: Dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)

        model_name_map = {
            'smolLM': 'HuggingFaceTB/SmolLM-135M',
            'smolLM360': 'HuggingFaceTB/SmolLM-360M',
            'smolLM1.7B': 'HuggingFaceTB/SmolLM-1.7B'
        }

        self.model_params = model_params
        self.model = AnoLLM(llm=model_name_map.get(self.model_params.model, self.model_params.model), 
                            epochs=model_params.epochs,
                            batch_size=data_params.batch_size, 
                            max_length_dict={col: 1000 for col in self.column_names})


    def train(self):
        self.model.fit(data=self.X_train, column_names=self.column_names)

    @torch.no_grad()
    def evaluate(self): 
        scores = self.model.decision_function(data=self.X_test, 
                                              column_names=self.column_names,
                                              n_permutations=self.model_params.n_permutations)
        metrics = get_summary_metrics(y_true=self.y_test, y_pred=scores.mean(axis=1))
        return metrics