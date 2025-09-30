
---
title: Home
layout: default
---

<div class="hero">
  <h2>ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection</h2>
  <p>A novel benchmark that restores textual semantics for tabular anomaly detection, enabling context-aware research and demonstrating substantial improvements through LLM-based approaches.</p>
  <div class="badges">
    <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper</a>
    <a href="https://github.com/yoonsanghyu/ReTabAD">💻 Code</a>
    <a href="{{ site.baseurl }}/dataset">🗂 Dataset</a>
  </div>
</div>

## TL;DR
ReTabAD addresses the fundamental disconnect in existing tabular anomaly detection benchmarks by restoring semantic context. It provides semantically-rich datasets with comprehensive metadata and demonstrates a **7.6% AUROC improvement** using LLMs with domain knowledge over context-agnostic approaches.

## Teaser
![teaser]({{ site.baseurl }}/assets/img/teaser.png)

## Features
- **📚 Semantically-Rich Datasets:** Raw data paired with comprehensive JSON metadata containing column descriptions, logical types, and characterizations
- **💡 Preserved Semantic Information:** Unified pipeline enabling fair comparisons across traditional ML, deep learning, and modern LLM approaches
- **🚀 Demonstrating LLM Potential:** Substantial 7.6% average AUROC improvement over context-agnostic models

## News
- 2025-09-30: Initial release.
