from addict import Dict
from retab.datasets import Preprocessor


class BaseTrainer:
    def __init__(self, data_params: Dict, model_params: Dict, preprocessor: Preprocessor, meta_info: Dict):
        self.data_params = data_params
        self.model_params = model_params
        self.preprocessor = preprocessor
        self.meta_info = meta_info

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
        
    def train(self):
        raise NotImplementedError()        

    def evaluate(self):
        raise NotImplementedError()