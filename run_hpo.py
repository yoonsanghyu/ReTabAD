import warnings; warnings.filterwarnings('ignore')
import os

import torch
import argparse
from dotenv import load_dotenv
import numpy as np; np.set_printoptions(suppress=True, precision=6)
import optuna

from retab.models import MODEL_LIST, TRAINERS
from retab.datasets import Preprocessor, DATA_LIST
from retab.utils import seed_everything, load_config, get_params, get_exp_id, args_to_dict, save_best_cfg_as_yaml


# load data directory configuration
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")


def objective(trial, cfg, meta_info):
    
    data_params = get_params(cfg, key="data_parameters", trial=trial, HPO=True)
    model_params = get_params(cfg, key="model_parameters", trial=trial, HPO=True)

    # assign ID to the experiment
    if meta_info.exp_id is None:
        meta_info.exp_id = get_exp_id(data_params, model_params)

    # load data using Preprocessor
    preprocessor = Preprocessor(
        serialize=data_params.serialize,
        ds_name=meta_info.data_name, 
        data_dir=DATA_DIR, 
        scaling_type=data_params.scaling_type, 
        cat_encoding=data_params.cat_encoding
    )

    # run experiments: multiple trials, report the mean.
    f1s = np.array([])
    aurocs = np.array([])
    auprcs = np.array([])

    try:
        # run over different seeds
        for seed in meta_info.seeds:
            # set seed
            seed_everything(seed)

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
            
            # log metrics
            for key, value in metrics.items():
                trial.set_user_attr(f"{key}_{seed}", value)
            
            f1s = np.append(f1s, metrics["f1"])
            aurocs = np.append(aurocs, metrics["auroc"])
            auprcs = np.append(auprcs, metrics["auprc"])
        
        # log final metrics
        trial.set_user_attr("f1s_mean", f1s.mean().item())
        trial.set_user_attr("aurocs_mean", aurocs.mean().item())
        trial.set_user_attr("auprcs_mean", auprcs.mean().item())
        trial.set_user_attr("f1s_std", f1s.std().item())
        trial.set_user_attr("aurocs_std", aurocs.std().item())
        trial.set_user_attr("auprcs_std", auprcs.std().item())
        return auprcs.mean().item()
    
    except Exception as e:
        print(f"Error: {e}")
        trial.set_user_attr("error", str(e))
        return -1


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Rethinking Anomaly Detection Benchmarks")
    parser.add_argument('--data_name', choices=DATA_LIST, type=str, default="wine", help=f"Specify the data name (CSV file without extension) from: {DATA_LIST}")
    parser.add_argument("--model_name", choices=MODEL_LIST, type=str, default="OCSVM", help=f"Specify the model name from: {MODEL_LIST}")
    parser.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4], help="List of random seeds to use (space separated, e.g., --seeds 1 2 3)")

    args = parser.parse_args()

    # meta info
    meta_info = args_to_dict(args)
    meta_info.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parameters for the model
    cfg = load_config(meta_info.cfg_file)

    # set search space
    search_space = {
        **{k: v['grid'] for k, v in cfg.hyperparameters.data_parameters.items()},
        **{k: v['grid'] for k, v in cfg.hyperparameters.model_parameters.items()},
    }
    study_name = f"{meta_info.model_name}_{meta_info.data_name}"
    study = optuna.create_study(
        direction=cfg.study.direction,
        storage=cfg.study.storage,
        study_name=study_name,
        sampler=optuna.samplers.GridSampler(search_space),     
    )
    
    study.optimize(lambda trial: objective(trial, cfg, meta_info), n_trials=cfg.study.n_trials)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
    best_cfg_file_path = os.path.join(cfg.exp.best_cfg_path, meta_info.data_name)
    os.makedirs(best_cfg_file_path, exist_ok=True)
    save_best_cfg_as_yaml(cfg, study.best_params, save_path=os.path.join(best_cfg_file_path, f"{meta_info.model_name}.yaml"))

    # save the result of best trial
    result_dir = os.path.join(cfg.exp.result_path, args.data_name, args.model_name)
    os.makedirs(result_dir, exist_ok=True)
    auprcs_mean, auprcs_std = study.best_trial.user_attrs["auprcs_mean"], study.best_trial.user_attrs["auprcs_std"]
    aurocs_mean, aurocs_std = study.best_trial.user_attrs["aurocs_mean"], study.best_trial.user_attrs["aurocs_std"]
    result_log_msg = f"auprc; {auprcs_mean} +- {auprcs_std} \n" + f"auroc; {aurocs_mean} +- {aurocs_std} \n"
    with open(os.path.join(result_dir, "result.txt"), 'w') as file:
        print(result_log_msg)
        file.write(result_log_msg)
    
    print("HPO Complete.")

if __name__ == "__main__":
    main()