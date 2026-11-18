# ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection

*Official code and data repository of ReTabAD: LLM-based Tabular Anomaly Detection Benchmark [ICLR 2026]*

[📄 Paper](https://arxiv.org/pdf/2510.02060) | [🌐 Project Page](https://yoonsanghyu.github.io/ReTabAD/) | [🤗 Dataset](https://huggingface.co/datasets/LGAI-DILab/ReTabAD)

Existing tabular anomaly detection benchmarks exhibit a fundamental disconnect from industrial practice. They systematically strip away semantic context by converting textual features into opaque encodings and discarding descriptive metadata. This prevents models from using the rich contextual information that domain experts routinely rely on, and that large language models (LLMs) can reason over.

`ReTabAD` closes this gap. It enables context-aware research by **<u>Re</u>storing** textual semantics for **<u>Tab</u>ular** **<u>A</u>nomaly** **<u>D</u>etection**.

## ✨ Key Features

* **📚 Semantically-Rich Datasets:** 20 datasets of raw data paired with JSON metadata containing column descriptions, logical types, and characterizations of normal data.
* **💡 One Pipeline for Every Paradigm:** A unified pipeline that preserves semantic information, enabling fair comparisons across traditional ML, deep learning, and LLM-based detectors.
* **🚀 LLMs as Zero-Shot Detectors:** With injected domain knowledge, LLMs improve average AUROC by **9.5%** (Gemini-2.5-Pro) over the same models on context-agnostic data that mimics traditional benchmarks.

## ⚙️ Installation

### 1. Clone the repository

```sh
git clone https://github.com/yoonsanghyu/ReTabAD.git
cd ReTabAD
```

### 2. Set up the environment

**Option A: pip**

<!-- TODO: requirements.txt가 있으면 아래 그대로, 없으면 만들어서 넣기 (pip 설치 경로가 있어야 처음 온 사람이 바로 써 봄) -->
```sh
pip install -r requirements.txt
```

**Option B: Docker** (requires `Docker` and the `NVIDIA Container Toolkit`)

```sh
docker build -t retabad:1.0.0 .
docker run -it --rm --gpus all -v $(pwd):/workspace/ReTabAD retabad:1.0.0
```
<!-- TODO: 컨테이너 안의 실제 작업 경로에 맞게 -v 경로 확인 -->

### 3. Download the data

The dataset is hosted on [Hugging Face](https://huggingface.co/datasets/LGAI-DILab/ReTabAD). Accept the access conditions on the dataset page, then:

```sh
huggingface-cli login
huggingface-cli download LGAI-DILab/ReTabAD --repo-type dataset --local-dir ./data
```

Each dataset is stored as `final_data/{dataset}/{dataset}.csv` (data) and `final_data/{dataset}/{dataset}.json` (semantic metadata).

Then create a `.env` file in the project root:

```bash
# .env
DATA_DIR=./data
```
<!-- TODO: 코드가 DATA_DIR 아래에서 final_data/를 찾는지, final_data 자체를 DATA_DIR로 기대하는지 확인 -->

## 🚀 How to Run

### 🧪 Quick Start: Run a Single Experiment

```sh
python run_default.py --data_name <DATASET_NAME> --model_name <MODEL_NAME> --cfg_file <CONFIG_FILE.yaml> [--seed 42]
```

- `<DATASET_NAME>`: one of the 20 datasets listed [below](#-benchmark-datasets).
- `<MODEL_NAME>`: one of the supported models (see `configs/default/`).
- `<CONFIG_FILE.yaml>`: path to the configuration YAML file.
- `--seed`: (optional) random seed (default: 42).

Classical detector:
```sh
python run_default.py --data_name wine --model_name OCSVM --cfg_file configs/default/pyod/OCSVM.yaml --seed 42
```

Zero-shot LLM detector with semantic context:
<!-- TODO: 실제 LLM 모델 이름과 config 경로로 교체 (LLM 예시가 있어야 LLM 연구자가 바로 따라 해 봄) -->
```sh
python run_default.py --data_name wine --model_name LLM_MODEL_NAME --cfg_file configs/default/llm/LLM_CONFIG.yaml --seed 42
```

### 📈 Hyperparameter Tuning

Hyperparameter search uses Optuna:

```sh
python run_hpo.py --data_name <DATASET_NAME> --model_name <MODEL_NAME> --cfg_file <CONFIG_FILE.yaml> [--seeds 0 1 2 3 4]
```

- `--seeds`: (optional) random seeds for repeated runs (default: 0 1 2 3 4).

Example:
```sh
python run_hpo.py --data_name wine --model_name OCSVM --cfg_file configs/hpo/pyod/OCSVM.yaml --seeds 0 1 2 3 4
```

To monitor runs with Optuna Dashboard:
```sh
mkdir exp_db
optuna-dashboard sqlite:///exp_db/retab.sqlite3
```

## 📊 Benchmark Datasets

ReTabAD includes **20 diverse datasets** spanning multiple domains:

| Dataset Name | Datapoints | Columns | Normal Count | Anomaly Count | Anomaly Ratio (%) |
| :--- | ---:| ---:| ---:| ---:| ---:|
| automobile | 159 | 25 | 117 | 42 | 26.42 |
| backdoor | 29,223 | 42 | 29,113 | 110 | 0.38 |
| campaign | 7,842 | 16 | 6,056 | 1,786 | 22.77 |
| cardiotocography | 2,126 | 21 | 1,655 | 471 | 22.15 |
| census | 50,000 | 41 | 47,121 | 2,879 | 5.76 |
| churn | 7,032 | 19 | 5,163 | 1,869 | 26.58 |
| cirrhosis | 247 | 17 | 165 | 82 | 33.20 |
| covertype | 50,000 | 12 | 49,520 | 480 | 0.96 |
| credit | 30,000 | 23 | 23,364 | 6,636 | 22.12 |
| equip | 7,672 | 6 | 6,905 | 767 | 10.00 |
| gallstone | 241 | 38 | 161 | 80 | 33.20 |
| glass | 214 | 9 | 163 | 51 | 23.83 |
| glioma | 730 | 23 | 487 | 243 | 33.29 |
| quasar | 50,000 | 8 | 40,520 | 9,480 | 18.96 |
| seismic | 2,584 | 18 | 2,414 | 170 | 6.58 |
| stroke | 4,909 | 10 | 4,700 | 209 | 4.26 |
| vertebral | 310 | 6 | 210 | 100 | 32.26 |
| wbc | 535 | 30 | 357 | 178 | 33.27 |
| wine | 178 | 13 | 130 | 48 | 26.97 |
| yeast | 1,484 | 8 | 1,389 | 95 | 6.40 |

## 🏆 Add Your Method

We welcome new results on ReTabAD. Open an issue or pull request with your method's results, or contact us at sanghyu.yoon@lgresearch.ai.

## 📖 Citation

If you find ReTabAD useful, please cite:

```bibtex
@inproceedings{yoon2026retabad,
  title     = {ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection},
  author    = {Yoon, Sanghyu and Kim, Dongmin and Yoon, Suhee and Sim, Ye Seul and Yoa, Seungdong and Cho, Hye-Seung and Lee, Soonyoung and Lee, Hankook and Lim, Woohyung},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=UFwgg44VZq}
}
```
