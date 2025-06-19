import os

import torch
import torch.optim as optim
from addict import Dict

from .Model import MCM
from .Loss import LossFunction
from .Score import ScoreFunction
from retab.utils import get_summary_metrics
from retab.datasets import Preprocessor
from retab.models import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, data_params: dict, model_params: dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)

        # prepare model parameters for MCM
        self.device = meta_info.device
        self.sche_gamma = model_params.sche_gamma
        self.learning_rate = model_params.learning_rate
        model_params.data_dim = self.preprocessor.nfeatures
        model_params.device = meta_info.device

        self.model = MCM(model_params).to(self.device)
        self.loss_fuc = LossFunction(model_params).to(self.device)
        self.score_func = ScoreFunction(model_params).to(self.device)

        # other parameters
        self.epochs = model_params.epochs
        self.ckpt_path = os.path.join(meta_info.checkpoint_path, meta_info.data_name, meta_info.model_name, meta_info.exp_id, f'{meta_info.seed}.pth')
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        min_loss = 1e9-1
        for epoch in range(self.epochs):
            for step, data in enumerate(self.trainloader):
                x_input = data["cont_features"].to(self.device)
                x_pred, z, masks = self.model(x_input)
                loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
            print(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
            if loss < min_loss:
                self.save()
                min_loss = loss
        print("Training complete.")

    @torch.no_grad()
    def evaluate(self):
        self.load()
        self.model.eval()
        mse_score, test_label = [], []
        for step, data in enumerate(self.testloader):
            x_input = data["cont_features"].to(self.device)
            x_pred, z, masks = self.model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(data["label"])
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        metrics = get_summary_metrics(y_true=test_label, y_pred=mse_score)
        return metrics
    
    def save(self):
        torch.save(self.model, self.ckpt_path)

    def load(self):
        self.model = torch.load(self.ckpt_path)
