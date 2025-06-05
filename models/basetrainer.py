from addict import Dict
from datasets import Preprocessor


class BaseTrainer:
    def __init__(self, data_params: Dict, model_params: Dict, preprocessor: Preprocessor, meta_info: Dict):
        self.data_params = data_params
        self.model_params = model_params
        self.preprocessor = preprocessor
        self.meta_info = meta_info

        self.train_dict, self.test_dict = self.preprocessor.prepare_data()

        self.X_train_cat, self.X_train_cont, self.y_train = (
            self.train_dict['X_cat_data'], self.train_dict['X_cont_data'], self.train_dict['y']
        )
        self.X_test_cat, self.X_test_cont, self.y_test = (
            self.test_dict['X_cat_data'], self.test_dict['X_cont_data'], self.test_dict['y']
        )
        
    def train(self):
        raise NotImplementedError()        

    def evaluate(self):
        raise NotImplementedError()