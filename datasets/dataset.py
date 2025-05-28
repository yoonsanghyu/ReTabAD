import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset

class Preprocessor:
    def __init__(self, ds_name, seed=42, task='anomaly',
                 scaling_type='standard', cat_encoding='int'):
        assert task == 'anomaly', "Only 'anomaly' task is supported."
        np.random.seed(seed)

        self.ds_name = ds_name
        self.scaling_type = scaling_type
        self.cat_encoding = cat_encoding

        self.data = pd.read_csv(f'./data/{ds_name}.csv')
        self.X = self.data.drop(columns=['label', 'prompt'], errors='ignore')
        self.y = np.array(self.data['label'], dtype=int)

        self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.continuous_columns = [col for col in self.X.columns if col not in self.categorical_columns]
        self.cat_dims = []

    def prepare_data(self):
        if self.cat_encoding == 'onehot':
            self._encode_onehot()
        elif self.cat_encoding == 'int':
            self._encode_int()
        elif self.cat_encoding == 'int_emb':
            self._encode_int_emb()
        else:
            raise NotImplementedError(f"Unsupported cat_encoding: {self.cat_encoding}")

        self._impute_and_cast()
        self.y = LabelEncoder().fit_transform(self.y)

        normal_idx = np.where(self.y == 0)[0]
        anomaly_idx = np.where(self.y == 1)[0]

        num_train = len(normal_idx) // 2
        train_idx = normal_idx[:num_train]
        test_idx = np.concatenate([normal_idx[num_train:], anomaly_idx])

        nan_mask = self.X.notnull().astype(int)

        X_train, y_train = self._split(self.X, self.y, nan_mask, train_idx)
        X_test, y_test = self._split(self.X, self.y, nan_mask, test_idx)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        self.cat_idxs, self.con_idxs = self._compute_feature_indices()

        # --- 6. Scaling Params ---
        self.scaling_params = self._compute_scaling_stats()

        return (
            self.cat_dims, self.cat_idxs, self.con_idxs,
            X_train, y_train, X_test, y_test,
            self.scaling_params
        )
    
    def _encode_onehot(self):
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        ohe_arr = ohe.fit_transform(self.X[self.categorical_columns])
        ohe_cols = ohe.get_feature_names(self.categorical_columns)
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=self.X.index)
        self.X = pd.concat([self.X.drop(columns=self.categorical_columns), df_ohe], axis=1)
        self.continuous_columns = self.X.columns.tolist()
        self.categorical_columns = []

    def _encode_int(self):
        for col in self.categorical_columns:
            self.X[col] = self.X[col].fillna("MissingValue")
            self.X[col] = LabelEncoder().fit_transform(self.X[col])
        self.continuous_columns = self.X.columns.tolist()
        self.categorical_columns = []

    def _encode_int_emb(self):
        for col in self.categorical_columns:
            self.X[col] = self.X[col].fillna("MissingValue")
            le = LabelEncoder().fit(self.X[col])
            self.X[col] = le.transform(self.X[col])
            self.cat_dims.append(len(le.classes_))

    def _impute_and_cast(self):
        for col in self.continuous_columns:
            self.X[col] = self.X[col].astype(float)
            self.X[col].fillna(self.X[col].mean(), inplace=True)

    def _split(self, X, y, mask, indices):
        x_data = {'data': X.values[indices], 'mask': mask.values[indices]}
        y_data = {'data': y[indices].reshape(-1, 1)}
        if x_data['data'].shape != x_data['mask'].shape:
            raise ValueError('Shape mismatch between data and mask!')
        return x_data, y_data

    def _compute_feature_indices(self):
        if self.cat_encoding in ['onehot', 'int']:
            return [], list(range(self.X.shape[1]))
        else:
            cat_idxs = [self.X.columns.get_loc(col) for col in self.categorical_columns]
            con_idxs = [self.X.columns.get_loc(col) for col in self.continuous_columns]
            return cat_idxs, con_idxs

    def _compute_scaling_stats(self):
        data = self.X_train['data']
        if self.cat_encoding == 'onehot':
            normalize_mask = np.zeros(data.shape[1], dtype=bool)
            for col in self.continuous_columns:
                if col in self.X.columns:
                    normalize_mask[self.X.columns.get_loc(col)] = True
            sub_data = data[:, normalize_mask]
        else:
            sub_data = data[:, self.con_idxs]

        if self.scaling_type == 'minmax':
            d_min = sub_data.min(0)
            d_max = sub_data.max(0)
            d_range = np.where(d_max - d_min < 1e-6, 1e-6, d_max - d_min)
            return {'type': 'minmax', 'min': d_min, 'range': d_range}

        elif self.scaling_type == 'standard':
            mean = sub_data.mean(0)
            std = sub_data.std(0)
            std = np.where(std < 1e-6, 1e-6, std)
            return {'type': 'standard', 'mean': mean, 'std': std}

        else:
            raise NotImplementedError(f"Unsupported scaling_type: {self.scaling_type}")


class TabularADDataset(Dataset):
    """
    A PyTorch Dataset for Tabular Anomaly Detection with both categorical and continuous features.

    Parameters:
    - X (dict): A dictionary with keys 'data' and 'mask' (numpy arrays).
    - Y (dict): A dictionary with key 'data' (numpy array).
    - cat_cols (list[int]): List of column indices for categorical features.
    - task (str): Task type, 'clf' or 'reg'.
    - scaling_stats (dict, optional): Dictionary of scaling parameters, must include 'type'.
    """

    def __init__(self, X, Y, cat_cols, task='clf', scaling_stats=None):
        X_data, X_mask = X['data'], X['mask']
        n_features = X_data.shape[1]

        cat_cols = list(cat_cols)
        con_cols = list(set(range(n_features)) - set(cat_cols))

        # Split features
        self.cat_data = X_data[:, cat_cols].astype(np.int64)       # Categorical
        self.con_data = X_data[:, con_cols].astype(np.float32)     # Continuous
        self.cat_mask = X_mask[:, cat_cols].astype(np.int64)
        self.con_mask = X_mask[:, con_cols].astype(np.int64)

        # Labels
        self.y = Y['data'] if task == 'clf' else Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)

        # Scaling
        if scaling_stats is not None:
            self._apply_scaling(self.con_data, scaling_stats)

    def _apply_scaling(self, con_data, scaling_stats):
        s_type = scaling_stats['type']
        if s_type == 'standard':
            con_data[:] = (con_data - scaling_stats['mean']) / scaling_stats['std']
        elif s_type == 'minmax':
            con_data[:] = (con_data - scaling_stats['min']) / scaling_stats['range']
        else:
            raise NotImplementedError(f"Unknown scaling type: {s_type}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return: (cat_features), con_features, label, cat_mask, con_mask
        cat_feat = np.concatenate((self.cls[idx], self.cat_data[idx]))
        cat_msk = np.concatenate((self.cls_mask[idx], self.cat_mask[idx]))
        return cat_feat, self.con_data[idx], self.y[idx], cat_msk, self.con_mask[idx]