import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset


class Preprocessor:
    def __init__(self, ds_name, data_dir, seed=42, task='anomaly',
                 scaling_type='standard', cat_encoding='int'):
        assert task == 'anomaly', "Only 'anomaly' task is supported."
        np.random.seed(seed)

        self.ds_name = ds_name
        self.scaling_type = scaling_type
        self.cat_encoding = cat_encoding

        self.data = pd.read_csv(os.path.join(data_dir, f'{ds_name}.csv'))
        self.X = self.data.drop(columns=['label'], errors='ignore')
        self.y = np.array(self.data['label'], dtype=int)

        self.categorical_columns, self.continuous_columns = self.infer_column_types(self.X)
        self.org_continuous_columns = self.continuous_columns.copy()
        self.cat_dims = []
    
    def infer_column_types(self, X: pd.DataFrame, count_threshold: int = 5):
        categorical_columns = []
        continuous_columns = []

        for col in X.columns:
            series = X[col]
            if series.dtype.name in ['object', 'category']:
                categorical_columns.append(col)
            elif pd.api.types.is_numeric_dtype(series) and series.nunique() <= count_threshold:
                categorical_columns.append(col)
            else:
                continuous_columns.append(col)

        return categorical_columns, continuous_columns

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

        self.scaling_params = self._compute_scaling_stats()

        train_dict = self._make_dataset(self.X_train, self.y_train)
        test_dict = self._make_dataset(self.X_test, self.y_test)

        return train_dict, test_dict

    
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
        # Only continuous value
        data = self.X_train['data'][:,self.con_idxs]
        n_features = data.shape[1]

        if self.scaling_type is None:
            return {'type': 'none'}
        
        if self.cat_encoding == 'onehot':
            normalize_mask = np.zeros(n_features, dtype=bool)
            for col in self.org_continuous_columns:
                if col in self.X.columns:
                    idx = self.X.columns.get_loc(col)
                    normalize_mask[idx] = True
        else:
            normalize_mask = np.ones(n_features, dtype=bool)

        sub_data = data[:, normalize_mask]

        if self.scaling_type == 'minmax':
            d_min = sub_data.min(0)
            d_max = sub_data.max(0)
            d_range = d_max - d_min
            d_range = np.where(d_range < 1e-6, 1.0, d_range)

            full_min = np.zeros(n_features, dtype=np.float32)
            full_range = np.ones(n_features, dtype=np.float32)
            full_min[normalize_mask] = d_min
            full_range[normalize_mask] = d_range

            return {'type': 'minmax', 'min': full_min, 'range': full_range}

        elif self.scaling_type == 'standard':
            mean = sub_data.mean(0)
            std = sub_data.std(0)
            std = np.where(std < 1e-6, 1.0, std)  # NaN


            full_mean = np.zeros(n_features, dtype=np.float32)
            full_std = np.ones(n_features, dtype=np.float32)
            full_mean[normalize_mask] = mean
            full_std[normalize_mask] = std

            return {'type': 'standard', 'mean': full_mean, 'std': full_std}

        else:
            raise NotImplementedError(f"Unsupported scaling_type: {self.scaling_type}")

    def _make_dataset(self, X, y):
        X_data, X_mask = X['data'], X['mask']
        y = y['data']
        con_data = X_data[:, self.con_idxs].astype(np.float32)
        con_mask = X_mask[:, self.con_idxs].astype(np.int64)
        cat_data = X_data[:, self.cat_idxs].astype(np.int64) if self.cat_idxs else None
        cat_mask = X_mask[:, self.cat_idxs].astype(np.int64) if self.cat_idxs else None

        if self.scaling_params['type'] == 'standard':
            con_data = (con_data - self.scaling_params['mean']) / self.scaling_params['std']
        elif self.scaling_params['type'] == 'minmax':
            con_data = (con_data - self.scaling_params['min']) / self.scaling_params['range']

        # Add dummy cls token to categorical features
        cls_token = np.zeros((y.shape[0], 1), dtype=np.int64)
        cls_mask = np.ones((y.shape[0], 1), dtype=np.int64)

        if cat_data is not None:
            cat_data = np.concatenate([cls_token, cat_data], axis=1)
            cat_mask = np.concatenate([cls_mask, cat_mask], axis=1)
        else:
            cat_data = cls_token
            cat_mask = cls_mask

        return {
            'X_cat_data': cat_data,
            'X_cat_mask': cat_mask,
            'X_cont_data': con_data,
            'X_cont_mask': con_mask,
            'y': y
        }


class TabularADDataset(Dataset):
    def __init__(self, X_cat_data, X_cat_mask, X_cont_data, X_cont_mask, y):
        self.X_cat_data = X_cat_data
        self.X_cat_mask = X_cat_mask
        self.X_cont_data = X_cont_data
        self.X_cont_mask = X_cont_mask
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'cat_features': self.X_cat_data[idx],
            'cat_mask': self.X_cat_mask[idx],
            'cont_features': self.X_cont_data[idx],
            'cont_mask': self.X_cont_mask[idx],
            'label': self.y[idx]
        }
