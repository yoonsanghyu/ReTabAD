import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from addict import Dict

# from utils import aucPerformance, F1Performance

from .Model import DRL
from retab.utils import get_summary_metrics
from retab.datasets import Preprocessor
from retab.models import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, data_params: dict, model_params: dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)
        
        # prepare model parameters for DRL
        self.device = meta_info.device
        self.sche_gamma = model_params['sche_gamma']
        self.learning_rate = model_params['learning_rate']
        model_params.data_dim = self.preprocessor.nfeatures
        model_params.device = meta_info.device
        
        self.model = DRL(model_params).to(self.device)
        
        # other parameters
        self.epochs = model_params.epochs

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        for epoch in range(self.epochs):
            running_loss = 0.0
            for step, data in enumerate(self.trainloader):
                x_input = data["cont_features"].to(self.device)
                
                # decomposition loss
                loss = self.model(x_input).mean()

                # alignment loss
                if self.model_params['input_info'] == True:
                    h = self.model.encoder(x_input)
                    x_tilde = self.model.decoder(h)
                    s_loss = F.cosine_similarity(x_tilde, x_input, dim=-1).mean() * (-1)
                    loss += self.model_params['input_info_ratio'] * s_loss

                # separation loss
                if self.model_params['cl'] == True:
                    h_ = F.softmax(self.model.phi(x_input), dim=1)
                    selected_rows = np.random.choice(h_.shape[0], int(h_.shape[0] * 0.8), replace=False)
                    h_ = h_[selected_rows]

                    matrix = h_ @ h_.T
                    mol = torch.sqrt(torch.sum(h_**2, dim=-1, keepdim=True)) @ torch.sqrt(torch.sum(h_.T**2, dim=0, keepdim=True))
                    matrix = matrix / mol
                    d_loss =  ((1 - torch.eye(h_.shape[0]).cuda()) * matrix).sum() /(h_.shape[0]) / (h_.shape[0])
                    loss += self.model_params['cl_ratio'] * d_loss
                
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.trainloader)
            print(info.format(epoch,running_loss))

        self.save()
        print("Training complete.")

    @torch.no_grad()
    def evaluate(self):
        self.load()
        self.model.eval()
        mse_score, test_label = [], []
        for step, data in enumerate(self.testloader):
            x_input = data["cont_features"].to(self.device)

            h = self.model.encoder(x_input)

            weight = F.softmax(self.model.phi(x_input), dim=1)
            h_ = weight@self.model.basis_vector

            mse = F.mse_loss(h, h_, reduction='none')
            mse_batch = mse.mean(dim=-1, keepdim=True)
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
