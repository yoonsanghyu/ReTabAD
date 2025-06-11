import warnings; warnings.filterwarnings('ignore')
import os
import json
import pandas as pd

import argparse
from dotenv import load_dotenv
import numpy as np; np.set_printoptions(suppress=True, precision=6)
import torch

from retab.datasets import Preprocessor, DATA_LIST
from retab.models import MODEL_LIST, TRAINERS
from retab.utils import seed_everything, load_config, get_params, get_exp_id, args_to_dict

# load data directory configuration
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Rethinking Anomaly Detection Benchmarks")
    parser.add_argument('--data_name', choices=DATA_LIST, type=str, default="wine", help=f"Specify the data name (CSV file without extension) from: {DATA_LIST}")
    parser.add_argument("--model_name", choices=MODEL_LIST, type=str, default="AnoLLM", help=f"Specify the model name from: {MODEL_LIST}")
    parser.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # seed
    seed_everything(args.seed)

    # meta info
    meta_info = args_to_dict(args)
    meta_info.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parameters for the model
    cfg = load_config(meta_info.cfg_file)
    data_params = get_params(cfg, key="data_parameters")
    model_params = get_params(cfg, key="model_parameters")

    # assign ID to the experiment
    if meta_info.exp_id is None:
        meta_info.exp_id = get_exp_id(data_params, model_params)

    # initialize preprocessor
    preprocessor = Preprocessor(
        serialize=data_params.serialize,
        ds_name=meta_info.data_name, 
        data_dir=DATA_DIR, 
        scaling_type=data_params.scaling_type, 
        cat_encoding=data_params.cat_encoding
    )

    # initialize trainer
    trainer = TRAINERS[meta_info.model_name](
        data_params=data_params,
        model_params=model_params,
        preprocessor=preprocessor,
        meta_info=meta_info
    )

    # train and evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    
    # save results
    result_path = os.path.join(cfg.exp.result_path, meta_info.data_name, meta_info.model_name)
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, f"{meta_info.exp_id}.json"), 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
