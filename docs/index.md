---
title: Home
layout: default
---

<div class="hero">
  <h2>🤔 ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection</h2>
  <p class="authors">Sanghyu Yoon, Dongmin Kim, Suhee Yoon, Ye Seul Sim, Seungdong Yoa, Hye-Seung Cho, Soonyoung Lee, Hankook Lee, Woohyung Lim</p>
  <div class="badges">
    <a href="https://arxiv.org/pdf/2510.02060">📄 Paper</a>
    <a href="https://github.com/yoonsanghyu/ReTabAD">💻 Code</a>
    <a href="https://huggingface.co/datasets/LGAI-DILab/ReTabAD">🗂 Dataset</a>
  </div>
</div>

## 🎯 Overview

<div style="text-align: center; margin: 2rem 0;">
  <img src="{{ site.baseurl }}/assets/img/overview.png" alt="ReTabAD Overview" style="width: 80%; max-width: 100%;">
</div>

In tabular anomaly detection (AD), textual semantics often carry critical signals, as the definition of an anomaly is closely tied to domain-specific context. However, existing benchmarks provide only raw data points without semantic context, overlooking rich textual metadata such as feature descriptions and domain knowledge that experts rely on in practice. This limitation restricts research flexibility and prevents models from fully leveraging domain knowledge for detection. ReTabAD addresses this gap by Restoring textual semantics to enable context-aware Tabular AD research. We provide (1) 20 carefully curated tabular datasets enriched with structured textual metadata, together with implementations of state-of-the-art AD algorithms—including classical, deep learning, and LLM-based approaches—and (2) a zero-shot LLM framework that leverages semantic context without task-specific training, establishing a strong baseline for future research. Furthermore, this work provides insights into the role and utility of textual metadata in AD through experiments and analysis. Results show that semantic context improves detection performance and enhances interpretability by supporting domain-aware reasoning. These findings establish ReTabAD as a benchmark for systematic exploration of context-aware AD.

---

## ✨ Key Features

<div class="grid">
  <div class="card">
    <h3>📚 Semantically-Rich Tabular AD Benchmark</h3>
    <p>Tabular data paired with comprehensive JSON text metadata containing column descriptions, logical types, and characterizations of normal data.</p>
  </div>
  <div class="card">
    <h3>💡 Support SOTA Algorithms</h3>
    <p>Unified pipeline enabling fair comparisons across traditional ML, deep learning, and modern LLM approaches.</p>
  </div>
  <div class="card">
    <h3>🚀 LLM Potential</h3>
    <p>Demonstrates substantial performance improvements when models can leverage semantic information.</p>
  </div>
</div>

---

## 🔬 Why ReTabAD?

Traditional tabular AD benchmarks exhibit a **fundamental disconnect from industrial practice**:

- ❌ **Lost Semantics**: Textual features converted to opaque encodings
- ❌ **Missing Context**: Descriptive metadata discarded
- ❌ **Rigid Preprocessing**: Limits research flexibility
- ❌ **No Domain Knowledge**: Prevents models from using expert insights

**ReTabAD solves these problems** by restoring semantic context and enabling context-aware AD research.

---

## 📊 Benchmark Statistics

ReTabAD includes **20 diverse datasets** spanning multiple domains:

| Metric | Range |
|--------|-------|
| **Datasets** | 20 real-world scenarios |
| **Data Points** | 159 - 50,000 per dataset |
| **Features** | 6 - 42 columns |
| **Anomaly Ratio** | 0.38% - 33.29% |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yoonsanghyu/ReTabAD.git
cd ReTabAD

# Build Docker image
docker build -t retabad:1.0.0 .

# Run experiment
python run_default.py --data_name wine --model_name OCSVM --cfg_file configs/default/pyod/OCSVM.yaml
```

See [Usage]({{ site.baseurl }}/usage) for detailed instructions.

---

## 📰 News

- **2025-09-30**: Initial release of ReTabAD benchmark
- **2025-09-30**: 20 datasets with semantic metadata published
- **2025-09-30**: GitHub repository and documentation launched
