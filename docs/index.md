---
title: Home
layout: default
---

<div class="hero">
  <h2>🤔 ReTabAD: Restoring Semantic Context in Tabular Anomaly Detection</h2>
  <p>A novel benchmark that restores textual semantics for tabular anomaly detection, enabling context-aware research and demonstrating substantial improvements through LLM-based approaches.</p>
  <div class="badges">
    <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper</a>
    <a href="https://github.com/yoonsanghyu/ReTabAD">💻 Code</a>
    <a href="{{ site.baseurl }}/dataset">🗂 Dataset</a>
  </div>
</div>

## 🎯 Overview

![ReTabAD Overview]({{ site.baseurl }}/assets/img/overview.png)

Existing tabular anomaly detection benchmarks **systematically strip away semantic context** by converting textual features into opaque encodings. ReTabAD addresses this fundamental disconnect by providing semantically-rich datasets with comprehensive metadata.

Our results show a **substantial average AUROC improvement of 9.5%** (Gemini-2.5-Pro) using LLMs with domain knowledge over context-agnostic approaches.

---

## ✨ Key Features

<div class="grid">
  <div class="card">
    <h3>📚 Semantically-Rich Datasets</h3>
    <p>Raw data paired with comprehensive JSON metadata containing column descriptions, logical types, and characterizations of normal data.</p>
  </div>
  <div class="card">
    <h3>💡 Preserved Context</h3>
    <p>Unified pipeline enabling fair comparisons across traditional ML, deep learning, and modern LLM approaches.</p>
  </div>
  <div class="card">
    <h3>🚀 LLM Potential</h3>
    <p>Demonstrates substantial performance improvements when models can leverage semantic information.</p>
  </div>
</div>

---

## 🔬 Why ReTabAD?

Traditional tabular anomaly detection benchmarks exhibit a **fundamental disconnect from industrial practice**:

- ❌ **Lost Semantics**: Textual features converted to opaque encodings
- ❌ **Missing Context**: Descriptive metadata discarded
- ❌ **Rigid Preprocessing**: Limits research flexibility
- ❌ **No Domain Knowledge**: Prevents models from using expert insights

**ReTabAD solves these problems** by restoring semantic context and enabling context-aware anomaly detection research.

---

## 📊 Benchmark Statistics

ReTabAD includes **20 diverse datasets** spanning multiple domains:

| Metric | Range |
|--------|-------|
| **Datasets** | 20 real-world scenarios |
| **Data Points** | 80 - 50,000 per dataset |
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
