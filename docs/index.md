---
title: Home
layout: default
---

<div class="hero">
  <h2> ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection</h2>
  <p class="authors">Sanghyu Yoon, Dongmin Kim, Suhee Yoon, Ye Seul Sim, Seungdong Yoa,<br>Hye-Seung Cho, Soonyoung Lee, Hankook Lee, Woohyung Lim</p>
  <div class="badges">
    <a href="https://arxiv.org/pdf/2510.02060">📄 Paper</a>
    <a href="https://github.com/yoonsanghyu/ReTabAD">
      <svg height="16" width="16" viewBox="0 0 16 16" style="vertical-align: middle; margin-right: 4px;">
        <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
      </svg>
      GitHub
    </a>
    <a href="https://huggingface.co/datasets/LGAI-DILab/ReTabAD">🗂 Dataset</a>
  </div>
</div>

## 🎯 Overview

<div style="text-align: center; margin: 2rem 0;">
  <img src="{{ site.baseurl }}/assets/img/overview.png" alt="ReTabAD Overview" style="width: 80%; max-width: 100%;">
</div>

<div style="text-align: justify;">
In tabular anomaly detection (AD), textual semantics often carry critical signals, as the definition of an anomaly is closely tied to domain-specific context. However, existing benchmarks provide only raw data points without semantic context, overlooking rich textual metadata such as feature descriptions and domain knowledge that experts rely on in practice. This limitation restricts research flexibility and prevents models from fully leveraging domain knowledge for detection. ReTabAD addresses this gap by Restoring textual semantics to enable context-aware Tabular AD research. We provide (1) 20 carefully curated tabular datasets enriched with structured textual metadata, together with implementations of state-of-the-art AD algorithms—including classical, deep learning, and LLM-based approaches—and (2) a zero-shot LLM framework that leverages semantic context without task-specific training, establishing a strong baseline for future research. Furthermore, this work provides insights into the role and utility of textual metadata in AD through experiments and analysis. Results show that semantic context improves detection performance and enhances interpretability by supporting domain-aware reasoning. These findings establish ReTabAD as a benchmark for systematic exploration of context-aware AD.
</div>



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



## 🔬 Why ReTabAD?

Traditional tabular AD benchmarks exhibit a **fundamental disconnect from industrial practice**:

- ❌ **Lost Semantics**: Textual features converted to opaque encodings
- ❌ **Missing Context**: Descriptive metadata discarded
- ❌ **Rigid Preprocessing**: Limits research flexibility
- ❌ **No Domain Knowledge**: Prevents models from using expert insights

**ReTabAD solves these problems** by restoring semantic context and enabling context-aware AD research.



## 📊 Benchmark Statistics

ReTabAD includes **20 diverse datasets** spanning multiple domains:

| Metric | Range |
|--------|-------|
| **Datasets** | 20 real-world scenarios |
| **Data Points** | 159 - 50,000 per dataset |
| **Features** | 6 - 42 columns |
| **Anomaly Ratio** | 0.38% - 33.29% |



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



## 📰 News

- **2025-09-30**: Initial release of ReTabAD benchmark
- **2025-09-30**: 20 datasets with semantic metadata published
- **2025-09-30**: GitHub repository and documentation launched


