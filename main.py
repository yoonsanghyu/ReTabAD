import warnings; warnings.filterwarnings('ignore')
import os
import json

import argparse
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

from utils import seed_everything; seed_everything(42)
from datasets.dataset import Preprocessor

np.set_printoptions(suppress=True, precision=6)

# load data directory configuration
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

# Argument parsing
parser = argparse.ArgumentParser(description="Run anomaly detection pipeline.")
parser.add_argument('--data_name', type=str, default='hepatitis', help='Name of the dataset (CSV file without extension)')
parser.add_argument('--preprocess', type=str, choices=['standard', 'minmax', None], default='standard', help='Preprocessing method: standard or minmax')
parser.add_argument('--model_name', type=str, choices=['IForest', 'KNN', 'LOF', 'OCSVM', 'PCA'], default='IForest', help='Model to use: iforest')
parser.add_argument('--random_state', type=int, default=0, help='Random state for train-test split')
parser.add_argument('--cat_encoding', type=str, choices=['int', 'onehot', 'int_emb'], default='int', help='Categorical encoding method: int, onehot, or int_emb')

args = parser.parse_args()

# load data using Preprocessor
preprocessor = Preprocessor(ds_name=args.data_name, 
                            data_dir=DATA_DIR, 
                            scaling_type=args.preprocess, 
                            cat_encoding=args.cat_encoding)
train_dict, test_dict = preprocessor.prepare_data()

X_train_cat, X_train_cont, y_train = (
    train_dict['X_cat_data'], train_dict['X_cont_data'], train_dict['y']
)
X_test_cat, X_test_cont, y_test = (
    test_dict['X_cat_data'], test_dict['X_cont_data'], test_dict['y']
)
# train model
MODELS = {
    "IForest": IForest(),
    "KNN": KNN(),
    "LOF": LOF(),
    "OCSVM": OCSVM(),
    "PCA": PCA(),
}
clf = MODELS[args.model_name]
clf.fit(X_train_cont)
y_pred = clf.decision_function(X_test_cont)

# evaluate model
auroc = roc_auc_score(y_test, y_pred)
auprc = average_precision_score(y_test, y_pred)
print(auroc)
print(auprc)

# Save results
result_dir = os.path.join("results", args.data_name)
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, f"{args.model_name}.txt")
with open(result_path, 'w') as f:
    f.write(f"auroc: {auroc}\n")
    f.write(f"auprc: {auprc}\n")
