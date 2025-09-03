# 🤔 ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection
Existing tabular anomaly detection benchmarks exhibit a fundamental disconnect from industrial practice. They systematically strip away semantic context by converting textual features into opaque encodings and discarding descriptive metadata. This preprocessing rigidity limits research flexibility and prevents models from leveraging rich contextual information that domain experts routinely use.

`ReTabAD` is a novel benchmark created to address this gap. It enables context-aware research by **<u>Re</u>storing** textual semantics for **<u>Tab</u>ular** **<u>A</u>nomaly** **<u>D</u>etection**.

## ✨ Key Features

* **📚 Semantically-Rich Datasets:** Provides raw data paired with comprehensive JSON metadata containing column descriptions, logical types, and characterizations of normal data.
* **💡 Preserved Semantic Information:** A unified pipeline that preserves semantic information, paving the way for fair comparisons across traditional ML, deep learning, and modern language model approaches.
* **🚀 Demonstrating LLM Potential:** We showcase ReTabAD's capabilities by evaluating Large Language Models as zero-shot anomaly detectors with injected domain knowledge. Our results show a substantial average AUROC improvement of **9.5%** (Gemini-2.5-Pro) over models operating on context-agnostic data, which mimics traditional benchmarks.

## ReTabAD Benchmark Datasets

| Dataset Name | Datapoints | Columns | Normal Count | Anomaly Count | Anomaly Ratio (%) |
| :--- | ---:| ---:| ---:| ---:| ---:|
| backdoor | 29,223 | 42 | 29,113 | 110 | 0.38 |
| campaign | 7,842 | 16 | 6,056 | 1,786 | 22.77 |
| cardiotocography | 2,126 | 21 | 1,655 | 471 | 22.15 |
| census | 50,000 | 41 | 47,121 | 2,879 | 5.76 |
| cirrhosis | 247 | 17 | 165 | 82 | 33.20 |
| covertype | 50,000 | 12 | 49,520 | 480 | 0.96 |
| credit | 30,000 | 23 | 23,364 | 6,636 | 22.12 |
| ecoli | 214 | 7 | 143 | 71 | 33.18 |
| equip | 7,672 | 6 | 6,905 | 767 | 10.00 |
| gallstone | 241 | 38 | 161 | 80 | 33.20 |
| glass | 214 | 9 | 163 | 51 | 23.83 |
| glioma | 730 | 23 | 487 | 243 | 33.29 |
| hepatitis | 80 | 19 | 67 | 13 | 16.25 |
| quasar | 50,000 | 8 | 40,520 | 9,480 | 18.96 |
| seismic | 2,584 | 18 | 2,414 | 170 | 6.58 |
| stroke | 4,909 | 10 | 4,700 | 209 | 4.26 |
| vertebral | 310 | 6 | 210 | 100 | 32.26 |
| wbc | 535 | 30 | 357 | 178 | 33.27 |
| wine | 178 | 13 | 130 | 48 | 26.97 |
| yeast | 1,484 | 8 | 1,389 | 95 | 6.40 |

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


### Prerequisites

This project requires `Docker` and the `NVIDIA Container Toolkit` to be installed on your system.

### ⚙️ Installation & Environment Setup

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/carrtesy/ReTabAD.git](https://github.com/carrtesy/ReTabAD.git)
    cd ReTabAD
    ```

2.  **Configure Data Directory**
    Create a `.env` file in the project root and add the following line to specify your data directory:
    ```bash
    # .env
    DATA_DIR=/path/to/your/data/dir
    ```

3.  **Build and Run the Docker Container**
    ```sh
    # Build the Docker image
    docker build -t retabad:1.0.0 .

    # Run the Docker container
    docker run -itd --rm --name retabad --gpus '"device=4,5,6,7"' -m 375g -v /:/workspace retabad:1.0.0
    ```

## 🚀 How to Run ?

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
