# 🤔 RETABAD: A BENCHMARK FOR RESTORING SEMANTIC CONTEXT IN TABULAR ANOMALY DETECTION
Existing tabular anomaly detection benchmarks exhibit a fundamental disconnect from industrial practice. They systematically strip away semantic context by converting textual features into opaque encodings and discarding descriptive metadata. This preprocessing rigidity limits research flexibility and prevents models from leveraging rich contextual information that domain experts routinely use.

`ReTabAD` is a novel benchmark created to address this gap. It enables context-aware research by **<u>Re</u>storing** textual semantics for **<u>Tab</u>ular** **<u>A</u>nomaly** **<u>D</u>etection**.

## ✨ Key Features

* **📚 Semantically-Rich Datasets:** Provides raw data paired with comprehensive JSON metadata containing column descriptions, logical types, and characterizations of normal data.
* **💡 Preserved Semantic Information:** A unified pipeline that preserves semantic information, paving the way for fair comparisons across traditional ML, deep learning, and modern language model approaches.
* **🚀 Demonstrating LLM Potential:** We showcase ReTabAD's capabilities by evaluating Large Language Models as zero-shot anomaly detectors with injected domain knowledge. Our results show a substantial average AUROC improvement of **9.5%** (Gemini-2.5-Pro) over models operating on context-agnostic data, which mimics traditional benchmarks.

## Environment setting

### Data Directory Configuration
Create a `.env` file in the project root and add the following line to specify your data directory:

```
# .env
DATA_DIR=/PATH/TO/YOUR/DATA/DIR
```

Replace the path with the location of your data if different.

### Docker
```sh
docker build -t retabad:1.0.0 .
docker run -itd --rm --name retabad --gpus '"device=4,5,6,7"' -m 375g -v /:/workspace retabad:1.0.0
```

## How to run this codes

### Run a Single Experiment
To run a single experiment with a specific configuration, use:

```sh
python run_default.py --data_name <DATASET_NAME> --model_name <MODEL_NAME> --cfg_file <CONFIG_FILE.yaml> [--seed 42]
```
- `<DATASET_NAME>`: Name of the dataset.
- `<MODEL_NAME>`: Name of the model.
- `<CONFIG_FILE.yaml>`: Path to the configuration YAML file.
- `--seed`: (Optional) Random seed (default: 42).

Example:
```sh
python run_default.py --data_name wine --model_name OCSVM --cfg_file configs/default/pyod/OCSVM.yaml --seed 42
```

### Hyperparameter Tuning
To perform hyperparameter tuning using Optuna, run the following command:

```sh
python run_hpo.py --data_name <DATASET_NAME> --model_name <MODEL_NAME> --cfg_file <CONFIG_FILE.yaml> [--seeds 0 1 2 3 4]
```
- `<DATASET_NAME>`: Name of the dataset (choose from the supported datasets).
- `<MODEL_NAME>`: Name of the model (choose from the supported models).
- `<CONFIG_FILE.yaml>`: Path to the configuration YAML file.
- `--seeds`: (Optional) List of random seeds for repeated experiments (default: 0 1 2 3 4).

Example:
```sh
python run_hpo.py --data_name wine --model_name OCSVM --cfg_file configs/hpo/pyod/OCSVM.yaml --seeds 0 1 2 3 4
```

### Optuna dashboard
If you want to monitor with optuna-dashboard, try:
```sh
mkdir exp_db
optuna-dashboard sqlite:///exp_db/retab.sqlite3
```
