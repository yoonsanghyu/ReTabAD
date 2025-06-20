import torch
import torch.optim as optim

from .Model import MCMNetwork
from .Loss import LossFunction
from .Score import ScoreFunction


class MCM:
    """Wrapper class to train and score the MCM model."""

    def __init__(self, model_params, device):
        self.device = device
        self.sche_gamma = model_params.sche_gamma
        self.learning_rate = model_params.learning_rate
        self.epochs = model_params.epochs

        model_params.data_dim = model_params.get('data_dim')
        model_params.device = device

        self.model = MCMNetwork(model_params).to(self.device)
        self.loss_fuc = LossFunction(model_params).to(self.device)
        self.score_func = ScoreFunction(model_params).to(self.device)

    def fit(self, dataloader):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        min_loss = 1e9 - 1
        for epoch in range(self.epochs):
            for data in dataloader:
                x_input = data["cont_features"].to(self.device)
                x_pred, z, masks = self.model(x_input)
                loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
            print(info.format(epoch, loss.cpu(), mse.cpu(), divloss.cpu()))
            if loss < min_loss:
                min_loss = loss
        print("Training complete.")

    @torch.no_grad()
    def decision_function(self, dataloader):
        self.model.eval()
        mse_score = []
        for data in dataloader:
            x_input = data["cont_features"].to(self.device)
            x_pred, z, masks = self.model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        return mse_score