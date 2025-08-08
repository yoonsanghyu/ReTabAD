"""
Data preprocessing module for tabular anomaly detection.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .data_utils import (
    infer_column_types,
    impute_and_cast,
    split_data,
    compute_feature_indices,
)


class Preprocessor:
    """
    Preprocessor for tabular data in anomaly detection tasks.
    
    Handles data loading, encoding, scaling, and splitting.
    """
    
    def __init__(
        self,
        serialize,
        ds_name,
        data_dir,
        seed: int = 42,
        task: str = "anomaly",
        scaling_type: str = "standard",
        cat_encoding: str = "int",
    ):
        assert task == "anomaly", "Only 'anomaly' task is supported."
        np.random.seed(seed)

        self.serialize = serialize
        self.ds_name = ds_name
        self.scaling_type = scaling_type
        self.cat_encoding = cat_encoding

        self.data = pd.read_csv(os.path.join(data_dir, ds_name, f"{ds_name}.csv"))
        self.X = self.data.drop(columns=["label"], errors="ignore")
        self.column_names = self.X.columns.tolist()
        self.y = np.array(self.data["label"], dtype=int)
        self.categorical_columns, self.continuous_columns = infer_column_types(self.X)
        self.org_continuous_columns = self.continuous_columns.copy()
        self.cat_dims = []

    def prepare_data(self):
        """
        Main method to prepare data for training and testing.
        
        Returns:
            tuple: (train_dict, test_dict) containing prepared data
        """
        if self.cat_encoding == "onehot":
            self._encode_onehot()
        elif self.cat_encoding == "int":
            self._encode_int()
        elif self.cat_encoding == "int_emb":
            self._encode_int_emb()
        elif self.cat_encoding == "txt_emb":
            self._encode_txt_emb()
        else:
            raise NotImplementedError(f"Unsupported cat_encoding: {self.cat_encoding}")

        self.X = impute_and_cast(
            self.X, self.categorical_columns, self.continuous_columns
        )
        self.y = LabelEncoder().fit_transform(self.y)

        normal_idx = np.where(self.y == 0)[0]
        anomaly_idx = np.where(self.y == 1)[0]

        np.random.shuffle(normal_idx)
        num_train = len(normal_idx) // 2
        train_idx = normal_idx[:num_train]
        test_idx = np.concatenate([normal_idx[num_train:], anomaly_idx])        

        nan_mask = self.X.notnull().astype(int)

        X_train, y_train = split_data(self.X, self.y, nan_mask, train_idx)
        X_test, y_test = split_data(self.X, self.y, nan_mask, test_idx)

        if self.serialize:
            train_dict = {
                'X_data': X_train['data'], 
                'y': y_train['data'],
                'column_names': self.column_names,
                'is_serialized': self.serialize
            }
            test_dict = {
                'X_data': X_test['data'],
                'y': y_test['data'],
                'column_names': self.column_names,
                'is_serialized': self.serialize
            }
            return train_dict, test_dict

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.nfeatures = self.X_train["data"].shape[1]

        self.cat_idxs, self.con_idxs = compute_feature_indices(
            self.X, self.cat_encoding, self.categorical_columns, self.continuous_columns
        )

        self.scaling_params = self._compute_scaling_stats()

        train_dict = self._make_dataset(self.X_train, self.y_train)
        test_dict = self._make_dataset(self.X_test, self.y_test)

        return train_dict, test_dict

    def _encode_onehot(self):
        """Apply one-hot encoding to categorical columns."""
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe_arr = ohe.fit_transform(self.X[self.categorical_columns])
        ohe_cols = ohe.get_feature_names_out(self.categorical_columns)
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=self.X.index)
        self.X = pd.concat(
            [self.X.drop(columns=self.categorical_columns), df_ohe], axis=1
        )
        self.continuous_columns = self.X.columns.tolist()
        self.categorical_columns = []

    def _encode_int(self):
        """Apply integer encoding to categorical columns."""
        for col in self.categorical_columns:
            self.X[col] = LabelEncoder().fit_transform(self.X[col])
        self.continuous_columns = self.X.columns.tolist()
        self.categorical_columns = []

    def _encode_int_emb(self):
        """Apply integer encoding with embedding dimensions tracking."""
        for col in self.categorical_columns:
            le = LabelEncoder().fit(self.X[col])
            self.X[col] = le.transform(self.X[col])
            self.cat_dims.append(len(le.classes_))

    def _encode_txt_emb(self):
        """Apply text embedding encoding (placeholder)."""
        pass

    def _compute_scaling_stats(self):
        """Compute scaling statistics for continuous features."""
        data = self.X_train["data"][:, self.con_idxs].astype(float)
        n_features = data.shape[1]

        if self.scaling_type is None:
            return {"type": "none"}

        if self.cat_encoding == "onehot":
            normalize_mask = np.zeros(n_features, dtype=bool)
            for col in self.org_continuous_columns:
                if col in self.X.columns:
                    idx = self.X.columns.get_loc(col)
                    normalize_mask[idx] = True
        else:
            normalize_mask = np.ones(n_features, dtype=bool)

        sub_data = data[:, normalize_mask]

        if self.scaling_type == "minmax":
            d_min = sub_data.min(0)
            d_max = sub_data.max(0)
            d_range = d_max - d_min
            d_range = np.where(d_range < 1e-6, 1.0, d_range)

            full_min = np.zeros(n_features, dtype=np.float32)
            full_range = np.ones(n_features, dtype=np.float32)
            full_min[normalize_mask] = d_min
            full_range[normalize_mask] = d_range

            return {"type": "minmax", "min": full_min, "range": full_range}

        elif self.scaling_type == "standard":
            mean = sub_data.mean(0)
            std = sub_data.std(0)
            std = np.where(std < 1e-6, 1.0, std)

            full_mean = np.zeros(n_features, dtype=np.float32)
            full_std = np.ones(n_features, dtype=np.float32)
            full_mean[normalize_mask] = mean
            full_std[normalize_mask] = std

            return {"type": "standard", "mean": full_mean, "std": full_std}

        elif self.scaling_type == "none":
            return {"type": "none"}

        else:
            raise NotImplementedError(f"Unsupported scaling_type: {self.scaling_type}")

    def _make_dataset(self, X, y):
        """Create dataset dictionary with processed features."""
        X_data, X_mask = X["data"], X["mask"]
        y = y["data"]

        # numerical value
        con_data = X_data[:, self.con_idxs].astype(np.float32)
        con_mask = X_mask[:, self.con_idxs].astype(np.int64)
        if self.scaling_params["type"] == "standard":
            con_data = (con_data - self.scaling_params["mean"]) / self.scaling_params[
                "std"
            ]
        elif self.scaling_params["type"] == "minmax":
            con_data = (con_data - self.scaling_params["min"]) / self.scaling_params[
                "range"
            ]

        # categorical value
        if self.cat_encoding == "txt_emb":
            cat_data = X_data[:, self.cat_idxs]
            cat_mask = X_mask[:, self.cat_idxs].astype(np.int64)

            cls_tokens = np.array(["CLS"] * y.shape[0], dtype=object).reshape(-1, 1)
            cls_masks = np.ones((y.shape[0], 1), dtype=np.int64)

            cat_data = np.concatenate([cls_tokens, cat_data], axis=1)
            cat_mask = np.concatenate([cls_masks, cat_mask], axis=1)

        elif self.cat_encoding == "int_emb":
            cat_data = X_data[:, self.cat_idxs].astype(np.int64)
            cat_mask = X_mask[:, self.cat_idxs].astype(np.int64)

            cls_token = np.zeros((y.shape[0], 1), dtype=np.int64)
            cls_mask = np.ones((y.shape[0], 1), dtype=np.int64)

            cat_data = np.concatenate([cls_token, cat_data], axis=1)
            cat_mask = np.concatenate([cls_mask, cat_mask], axis=1)
        else:
            cat_data = None
            cat_mask = None

        return {
            "X_cat_data": cat_data,
            "X_cat_mask": cat_mask,
            "X_cont_data": con_data,
            "X_cont_mask": con_mask,
            "y": y,
            "is_serialized": self.serialize,
        }