import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any


def infer_column_types(X: pd.DataFrame, count_threshold: int = 5) -> Tuple[List[str], List[str]]:
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


def impute_and_cast(X: pd.DataFrame, categorical_columns: List[str], 
                   continuous_columns: List[str]) -> pd.DataFrame:
    X_copy = X.copy()
    
    for col in continuous_columns:
        X_copy[col] = X_copy[col].astype(float)
        X_copy[col].fillna(X_copy[col].mean(), inplace=True)
    
    for col in categorical_columns:
        X_copy[col] = X_copy[col].astype(str)
        X_copy[col].fillna("MissingValue", inplace=True)
    
    return X_copy


def split_data(X: pd.DataFrame, y: np.ndarray, mask: pd.DataFrame, 
               indices: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    x_data = {'data': X.values[indices], 'mask': mask.values[indices]}
    y_data = {'data': y[indices].reshape(-1, 1)}
    
    if x_data['data'].shape != x_data['mask'].shape:
        raise ValueError('Shape mismatch between data and mask!')
    
    return x_data, y_data


def compute_feature_indices(X: pd.DataFrame, cat_encoding: str, 
                           categorical_columns: List[str], 
                           continuous_columns: List[str]) -> Tuple[List[int], List[int]]:
    if cat_encoding in ['onehot', 'int']:
        return [], list(range(X.shape[1]))
    else:
        cat_idxs = [X.columns.get_loc(col) for col in categorical_columns]
        con_idxs = [X.columns.get_loc(col) for col in continuous_columns]
        return cat_idxs, con_idxs

