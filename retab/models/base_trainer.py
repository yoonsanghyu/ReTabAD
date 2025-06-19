import os
from addict import Dict
from torch.utils.data import DataLoader

from retab.datasets import Preprocessor, TabularDataset


class BaseTrainer:
    def __init__(self, data_params: Dict, model_params: Dict, preprocessor: Preprocessor, meta_info: Dict):
        # initialize parameters
        self.data_params = data_params
        self.model_params = model_params
        self.preprocessor = preprocessor
        self.meta_info = meta_info

        # prepare data
        self.train_dict, self.test_dict = self.preprocessor.prepare_data()

        self.y_train = self.train_dict['y']
        self.y_test = self.test_dict['y']
        self.is_serialized = self.train_dict['is_serialized']

        if self.is_serialized:
            self.X_train = self.train_dict['X_data']
            self.X_test = self.test_dict['X_data']
            self.column_names = self.train_dict['column_names']
            self.X_train_cat = self.X_train_cont = None
            self.X_test_cat = self.X_test_cont = None
        else:
            self.X_train_cat = self.train_dict['X_cat_data']
            self.X_train_cont = self.train_dict['X_cont_data']
            self.X_test_cat = self.test_dict['X_cat_data']
            self.X_test_cont = self.test_dict['X_cont_data']
            self.X_train = self.X_test = None

        # pytorch dataset and dataloader
        self.trainset = TabularDataset(
            X_cat_data = self.train_dict["X_cat_data"],
            X_cat_mask = self.train_dict["X_cat_mask"],
            X_cont_data = self.train_dict["X_cont_data"],
            X_cont_mask = self.train_dict["X_cont_mask"],
            y = self.train_dict["y"]
        )
        self.trainloader = DataLoader(self.trainset, batch_size=self.data_params.batch_size, shuffle=True)

        self.testset = TabularDataset(
            X_cat_data = self.test_dict["X_cat_data"],
            X_cat_mask = self.test_dict["X_cat_mask"],
            X_cont_data = self.test_dict["X_cont_data"],
            X_cont_mask = self.test_dict["X_cont_mask"],
            y = self.test_dict["y"]
        )
        self.testloader = DataLoader(self.testset, batch_size=self.data_params.batch_size, shuffle=False)

        # checkpoint path for saving and loading model
        self.ckpt_path = os.path.join(meta_info.checkpoint_path, meta_info.data_name, meta_info.model_name, meta_info.exp_id, f'{meta_info.seed}.pth')
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

    def train(self):
        raise NotImplementedError()        

    def evaluate(self):
        raise NotImplementedError()
    
    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()