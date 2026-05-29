# Detecting Hope Across Languages: Multiclass Classification for Positive Online Discourse

This repository contains the official codebase and benchmarks for the paper **"Detecting Hope Across Languages: Multiclass Classification for Positive Online Discourse"**, developed for the **PolyHope-M 2025 shared task**.

This project shifts the digital content moderation paradigm from negative/toxic speech filtering toward the proactive identification and promotion of encouraging online discourse. It features a robust cross-lingual framework built on **XLM-RoBERTa** to classify fine-grained dimensions of hope in **English, Spanish, German, and Urdu** social media texts.

---

##  Abstract

The detection of hopeful speech in social media has emerged as a critical task for promoting positive discourse and well-being. In this paper, we present a machine learning approach to multiclass hope speech detection across multiple languages, including English, Urdu, and Spanish. We leverage transformer-based models, specifically XLM-RoBERTa, to detect and categorize hope speech into three distinct classes: Generalized Hope, Realistic Hope, and Unrealistic Hope. Our proposed methodology is evaluated on the PolyHope dataset for the PolyHope-M 2025 shared task, achieving competitive performance across all languages. We compare our results with existing models, demonstrating that our approach significantly outperforms prior state-of-the-art techniques in terms of macro F1 scores. We also discuss the challenges in detecting hope speech in low-resource languages and the potential for improving generalization. This work contributes to the development of multilingual, fine-grained hope speech detection models, which can be applied to enhance positive content moderation and foster supportive online communities.

---

##  Multi-Class Taxonomy

The system categorizes social media data using a two-level, fine-grained hierarchy to capture nuanced intent:

1. **Not Hope:** Toxic, neutral, or non-encouraging discourse (the dominant class across all language splits).
2. **Generalized Hope:** Broadly optimistic expressions not tied to specific actions or events (*e.g., "Better days are coming"*).
3. **Realistic Hope:** Encouragement grounded in plausible, actionable, or attainable outcomes.
4. **Unrealistic Hope:** Highly improbable or wishful positive assertions lacking rational grounding.

---

##  Methodology Architecture

The core pipeline leverages advanced transfer learning and iterative sample optimization to combat intense class imbalances across target languages:

* **Multilingual Transformer:** Fine-tuned **XLM-RoBERTa** (`xlm-roberta-base` / `xlm-roberta-large`) to capture shared semantic sub-word cues across diverse script geographies (Latin vs. Arabic/Nastaliq script for Urdu).
* **Baseline Comparison:** Feature extraction using TF-IDF text representations coupled with optimized **Logistic Regression (LR)** models.
* **Active Learning Loop:** Implements an iterative optimization strategy that identifies low-confidence model boundaries near underrepresented classes, helping sample and refine training distributions to maximize macro F1 performance.

---

##  Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* Hugging Face Transformers & Tokenizers
* Scikit-Learn
* Active-learning frameworks (*e.g., small-text or modAL*)

### Installation

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt

```

### Usage

1. **Text Preprocessing (Handling scripts, removing mentions, handles):**
```bash
python preprocess.py --languages en es ur de

```



```
2. **Run Baseline Classifiers:**
   ```bash
   python train_baseline.py --model logistic_regression

```

3. **Fine-tune Multilingual XLM-RoBERTa via Active Learning:**
```bash
python train_xlmr.py --epochs 4 --batch_size 16 --active_learning True

```



```
4. **Evaluate Multi-Class Macro Metrics:**
   ```bash
   python evaluate.py --metric macro_f1

```

---

##  Evaluation Results

Our approach achieves state-of-the-art results on the PolyHope-M 2025 benchmark data. Due to severe class disparities (with *Not Hope* skewing sample counts, particularly in English and Urdu), performance is strictly evaluated using **Macro F1-Scores** to guarantee high precision and recall on low-resource target frames.

---

##  Citation

If you utilize this cross-lingual architecture or reference the PolyHope-M 2025 baselines, please cite our work:

```bibtex
@article{faniyi2025detecting,
  title={Detecting Hope Across Languages: Multiclass Classification for Positive Online Discourse},
  author={Your Name and Co-Authors},
  journal={arXiv preprint arXiv:2509.25752},
  year={2025}
}

```

---
