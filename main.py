import os
import json

import argparse
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

from utils import seed_everything; seed_everything(42)


# Argument parsing
parser = argparse.ArgumentParser(description="Run anomaly detection pipeline.")
parser.add_argument('--data_name', type=str, default='cardiotocography', help='Name of the dataset (CSV file without extension)')
parser.add_argument('--preprocess', type=str, choices=['standard', 'minmax'], default='standard', help='Preprocessing method: standard or minmax')
parser.add_argument('--model', type=str, choices=['IForest', 'KNN', 'LOF', 'OCSVM', 'PCA'], default='IForest', help='Model to use: iforest')
parser.add_argument('--random_state', type=int, default=0, help='Random state for train-test split')
args = parser.parse_args()

data_name = args.data_name
preprocess = args.preprocess
model_name = args.model

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

# load data
df = pd.read_csv(os.path.join(DATA_DIR, f"{data_name}.csv"))
# Drop rows with missing values and print how many were dropped
df_before = len(df)
df = df.dropna()
df_after = len(df)
print(f"Dropped {df_before - df_after} records due to missing values.")

inliers = df[df['label'] == 0]
outliers = df[df['label'] == 1]
inliers_train, inliers_test = train_test_split(inliers, test_size=0.3, random_state=0) # todo: multiple random state

train_set = inliers_train
test_set = pd.concat([inliers_test, outliers], ignore_index=True)
X_train, y_train = train_set.loc[:, train_set.columns != "label"], train_set["label"]
X_test, y_test = test_set.loc[:, test_set.columns != "label"], test_set["label"]

# preprocess data

# Detect categorical columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

if len(cat_cols) > 0:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])
    # Print mapping as JSON
    mapping = {}
    for idx, col in enumerate(cat_cols):
        mapping[col] = {str(cls): int(i) for i, cls in enumerate(oe.categories_[idx])}
        mapping[col]['__unknown__'] = -1
    print("OrdinalEncoder Mappings:")
    print(json.dumps(mapping, indent=2, ensure_ascii=False))

if preprocess == 'standard':
    scaler = StandardScaler()
elif preprocess == 'minmax':
    scaler = MinMaxScaler()
else:
    raise ValueError(f"Unknown preprocess method: {preprocess}")

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
MODELS = {
    "IForest": IForest(),
    "KNN": KNN(),
    "LOF": LOF(),
    "OCSVM": OCSVM(),
    "PCA": PCA(),
}
clf = MODELS[model_name]
clf.fit(X_train)
y_pred = clf.decision_function(X_test)

# evaluate model
auroc = roc_auc_score(y_test, y_pred)
auprc = average_precision_score(y_test, y_pred)
print(auroc)
print(auprc)

# Save results
result_dir = os.path.join("results", data_name)
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, f"{model_name}.txt")
with open(result_path, 'w') as f:
    f.write(f"auroc: {auroc}\n")
    f.write(f"auprc: {auprc}\n")
