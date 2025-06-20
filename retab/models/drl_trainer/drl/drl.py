import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .Model import DRLNetwork

class DRL:
    """Wrapper class to train and score the DRL model."""

    def __init__(self, model_params, device):
        self.device = device
        self.sche_gamma = model_params.sche_gamma
        self.learning_rate = model_params.learning_rate
        self.epochs = model_params.epochs

        model_params.data_dim = model_params.get('data_dim')
        model_params.device = device
        self.model_params = model_params

        self.model = DRLNetwork(model_params).to(self.device)

    def fit(self, dataloader):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        for epoch in range(self.epochs):
            running_loss = 0.0
            for data in dataloader:
                x_input = data["cont_features"].to(self.device)

                # decomposition loss
                loss = self.model(x_input).mean()

                # alignment loss
                if self.model_params['input_info'] is True:
                    h = self.model.encoder(x_input)
                    x_tilde = self.model.decoder(h)
                    s_loss = F.cosine_similarity(x_tilde, x_input, dim=-1).mean() * (-1)
                    loss += self.model_params['input_info_ratio'] * s_loss

                # separation loss
                if self.model_params['cl'] is True:
                    h_ = F.softmax(self.model.phi(x_input), dim=1)
                    selected_rows = np.random.choice(h_.shape[0], int(h_.shape[0] * 0.8), replace=False)
                    h_ = h_[selected_rows]

                    matrix = h_ @ h_.T
                    mol = torch.sqrt(torch.sum(h_**2, dim=-1, keepdim=True)) @ torch.sqrt(torch.sum(h_.T**2, dim=0, keepdim=True))
                    matrix = matrix / mol
                    d_loss = ((1 - torch.eye(h_.shape[0]).to(self.device)) * matrix).sum() /(h_.shape[0]) / (h_.shape[0])
                    loss += self.model_params['cl_ratio'] * d_loss

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(dataloader)
            print(info.format(epoch, running_loss))
        print("Training complete.")

    @torch.no_grad()
    def decision_function(self, dataloader):
        self.model.eval()
        mse_score = []
        for data in dataloader:
            x_input = data["cont_features"].to(self.device)

            h = self.model.encoder(x_input)

            weight = F.softmax(self.model.phi(x_input), dim=1)
            h_ = weight @ self.model.basis_vector

            mse = F.mse_loss(h, h_, reduction='none')
            mse_batch = mse.mean(dim=-1, keepdim=True)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        return mse_score