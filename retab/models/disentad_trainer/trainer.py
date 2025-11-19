import os
import numpy as np
import torch
from addict import Dict

from .dis_net import DisNet
from retab.utils import get_summary_metrics
from retab.datasets import Preprocessor
from retab.models import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, data_params: dict, model_params: dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)
        
        self.device = meta_info.device
        model_params.data_dim = self.preprocessor.nfeatures

        self.model = DisNet(
            dim=model_params.hidden_dim, 
            att_dim=model_params.patch_size, 
            num_heads=self.model_params.num_heads, 
            qkv_bias=self.model_params.qkv_bias
        ).to(self.device)

        self.ckpt_path = os.path.join(meta_info.checkpoint_path, meta_info.data_name, meta_info.model_name, meta_info.exp_id, f'{meta_info.seed}.pth')
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_params.learning_rate)
        min_loss = np.inf
        for epoch in range(self.model_params.epochs):
            self.model.train()
            total_loss = 0
            reconstruction_loss = 0
            disentangle_loss = 0
            batch_idx = -1
            for i, sample in enumerate(self.trainloader):
                batch_idx += 1
                data = sample["cont_features"].to(self.device).unsqueeze(-1)
                recon_loss, dis_loss = self.model(data)

                reconstruction_loss += recon_loss.item()
                disentangle_loss += dis_loss.item()
                loss = recon_loss + dis_loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss = total_loss / (batch_idx + 1)
            reconstruction_loss = reconstruction_loss / (batch_idx + 1)
            disentangle_loss = disentangle_loss / (batch_idx + 1)
            print(f'Epoch: {epoch}, Total Loss: {total_loss}, Reconstruction Loss: {reconstruction_loss}, Disentangle Loss: {disentangle_loss}')
            if total_loss < min_loss:
                self.save()
                min_loss = total_loss
        
    @torch.no_grad()
    def evaluate(self):
        self.load()
        self.model.eval()
        scores = []
        for i, sample in enumerate(self.testloader):
            data = sample["cont_features"].to(self.device).unsqueeze(-1)
            anomaly_score = self.model(data).cpu().data.numpy().tolist()
            scores += anomaly_score
        scores = np.array(scores)
        metrics = get_summary_metrics(y_true=self.y_test, y_pred=scores)
        return metrics
    
    def save(self):
        torch.save(self.model, self.ckpt_path)

    def load(self):
        self.model = torch.load(self.ckpt_path)