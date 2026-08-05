<div align="center">

# 🗣️ Bangla → Chittagonian Dialect Translation
### Parameter-Efficient Fine-Tuning of Small Language Models with QLoRA

*A B.Sc. thesis project on low-resource dialect translation for Bangladesh's regional languages*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow?style=flat-square)](https://huggingface.co/)
[![QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA-8A2BE2?style=flat-square)](https://arxiv.org/abs/2305.14314)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#license)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)]()

[Overview](#-overview) •
[Models](#-models) •
[Dataset](#-dataset) •
[Methodology](#-methodology) •
[Results](#-results) •
[Setup](#-installation) •
[Usage](#-usage)

</div>

---

## 📖 Overview

**Chittagonian** is spoken by tens of millions of people in southeastern Bangladesh, yet it remains a low-resource dialect with almost no dedicated NLP tooling. This project investigates whether **small, instruction-tuned language models** — fine-tuned with **QLoRA** — can learn accurate Bangla → Chittagonian translation from a modest, manually-curated parallel corpus, under realistic, constrained compute (a single Colab T4 GPU).

The work covers the full pipeline: dataset construction, preprocessing, prompt design, multi-model QLoRA fine-tuning, and rigorous quantitative evaluation.

### Research Questions

| # | Question |
|---|----------|
| 1 | Can small language models be effectively fine-tuned for Bangla → Chittagonian translation? |
| 2 | Does QLoRA preserve translation quality while cutting training cost? |
| 3 | Which base model generalizes best under an identical training regime? |
| 4 | How well do standard MT metrics (BLEU, chrF++, ROUGE) capture quality in a low-resource dialect setting? |

---

## 🖼️ Thesis Defense

<div align="center">
<img src="sihabsafin.png" alt="Sihabul Islam Safin defending his B.Sc. thesis" width="520"/>

<sub>Presenting the final thesis at BGC Trust University Bangladesh</sub>
</div>

---

## 🧠 Models

Three instruction-tuned small language models were fine-tuned and evaluated under identical conditions for a fair, controlled comparison:

| Model | Parameters | Source |
|---|---|---|
| **Qwen2.5-3B-Instruct** | 3B | Alibaba / Qwen |
| **Gemma-2B-IT** | 2B | Google |
| **Llama-3.2-3B-Instruct** | 3B | Meta |

---

## 📊 Dataset

A Bangla–Chittagonian parallel corpus was built specifically for this thesis, as no adequate public dataset existed.

| Property | Value |
|---|---|
| Sentence pairs | **7,665** |
| Language pair | Bangla → Chittagonian |
| Sources | Facebook groups, native speakers, online forums, social media, community contributions |
| Split | 70% train / 15% validation / 15% test |
| Format | Instruction-style prompt pairs |

All pairs were manually reviewed for quality before being used in training.

---

## ⚙️ Methodology

```
Data Collection → Cleaning & Preprocessing → Train/Val/Test Split
      → Prompt Engineering → QLoRA Fine-Tuning (×3 models)
      → Inference → Evaluation (BLEU / chrF++ / ROUGE)
```

1. **Dataset Collection** — Gathered Bangla–Chittagonian sentence pairs from multiple community sources.
2. **Cleaning & Preprocessing** — Deduplication, missing-value handling, text normalization.
3. **Splitting** — 70/15/15 train/validation/test.
4. **Prompt Engineering** — Converted pairs into instruction-tuned prompt format.
5. **QLoRA Fine-Tuning** — 4-bit NF4 quantization with LoRA adapters; base weights frozen, only adapter parameters trained.
6. **Generation** — Produced Chittagonian translations from held-out Bangla inputs.
7. **Evaluation** — Scored with BLEU, chrF++, ROUGE-1, ROUGE-2, and ROUGE-L.

### Why QLoRA?

QLoRA enables fine-tuning of billion-parameter models on a single consumer-grade GPU by combining 4-bit quantization with low-rank adapters — freezing the base model and training only a small set of adapter weights. This made it possible to fine-tune three separate 2–3B models end-to-end within Colab's free-tier constraints, without sacrificing much translation quality.

---

## 🏆 Results

All three models were trained and evaluated on the identical dataset split and QLoRA configuration (rank 16, alpha 32).

| Model | BLEU | chrF++ | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|:---:|:---:|:---:|:---:|:---:|
| **Qwen2.5-3B-Instruct** 🥇 | **0.4179** | 12.72 | 0.0 | 0.0 | 0.0 |
| Gemma-2B-IT | 0.2712 | 14.23 | 0.0 | 0.0 | 0.0 |
| **Llama-3.2-3B-Instruct** 🥇 | 0.3755 | **16.89** | 0.0 | 0.0 | 0.0 |

<div align="center">
<img src="assets/model_comparison.png" alt="Model comparison bar chart across BLEU, chrF++, and ROUGE metrics" width="700"/>
</div>

**Key observations:**
- **Qwen2.5-3B-Instruct** scored highest on **BLEU** (0.4179), indicating stronger exact n-gram overlap with reference translations.
- **Llama-3.2-3B-Instruct** scored highest on **chrF++** (16.89), a character-level metric that tends to be more forgiving of morphological variation — relevant for a dialect with non-standardized spelling.
- **ROUGE-1/2/L scored 0.0 across all three models.** This is a known limitation when applying ROUGE — designed for summarization-style word/sequence overlap — to short, single-sentence dialect translations with high lexical divergence from standard Bangla. BLEU and chrF++ are more appropriate metrics for this task, and are weighted accordingly in the analysis.
- No single model dominates across every metric, underscoring that model choice for low-resource dialect translation depends on which quality dimension (exact-match precision vs. character-level fidelity) matters more for the downstream use case.

### Fine-Tuning Efficiency

| Model | LoRA Rank (r) | LoRA Alpha | Target Modules | Trainable Params (%) | Training Time (min) |
|---|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-3B-Instruct | 16 | 32 | 7 | 1.73% | 27.0 |
| Gemma-2B-IT | 16 | 32 | 4 | N/A | N/A |
| Llama-3.2-3B-Instruct | 16 | 32 | 4 | N/A | N/A |

> Efficiency logging was only fully captured for the Qwen2.5 run; Gemma and Llama were trained under the same QLoRA configuration but efficiency metrics were not recorded for those runs.

---

## 🗂️ Repository Structure

```text
Bangla-NLP/
├── data/
│   ├── raw/                # Original, unprocessed collected data
│   ├── interim/             # Partially cleaned intermediate data
│   ├── processed/           # Final, model-ready datasets
│   └── external/            # Any third-party reference data
├── outputs/
│   ├── evaluation/          # Raw evaluation results (BLEU, chrF++, ROUGE)
│   ├── figures/              # Plots and visualizations
│   ├── tables/                # Summary result tables
│   ├── reports/               # Thesis-related reports
│   └── models/                 # Saved QLoRA adapter checkpoints
├── src/
│   ├── data/                 # Dataset loading utilities
│   ├── eda/                   # Exploratory data analysis
│   ├── preprocessing/         # Cleaning & normalization scripts
│   ├── features/               # Prompt formatting / feature engineering
│   ├── models/                  # Model + QLoRA configuration
│   ├── training/                 # Fine-tuning pipelines
│   ├── evaluation/                # Metric computation scripts
│   ├── explainability/             # Analysis of model outputs
│   └── utils/                       # Shared helper functions
├── notebooks/                # Colab / Jupyter notebooks
├── assets/
│   └── model_comparison.png  # Evaluation results chart
├── sihabsafin.png             # Thesis defense photo
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

<div align="left">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/-PEFT-8A2BE2?style=flat-square)
![BitsAndBytes](https://img.shields.io/badge/-BitsAndBytes-333333?style=flat-square)
![TRL](https://img.shields.io/badge/-TRL-FF6F00?style=flat-square)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square)
![Google Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

</div>

---

## 🚀 Installation

```bash
git clone https://github.com/sihabsafin/bangla-chittagonian-translation
cd Bangla-NLP
pip install -r requirements.txt
```

> Running on Google Colab? Open the notebooks in `notebooks/` and make sure a GPU runtime is enabled (`Runtime → Change runtime type → GPU`).

---

## ▶️ Usage

**1. Prepare the dataset**
Ensure the cleaned/formatted dataset is placed in `data/processed/`.

**2. Preprocess**
```bash
python src/preprocessing/clean_data.py
```

**3. Format prompts**
```bash
python src/features/build_prompts.py
```

**4. Fine-tune with QLoRA**
```bash
python src/training/train_qlora.py --model qwen2.5-3b
```

**5. Evaluate**
```bash
python src/evaluation/evaluate.py --metrics bleu chrf rouge
```

---

## 🔭 Future Work

- Expand the parallel corpus with more dialectal variation and speakers
- Incorporate human evaluation alongside automatic metrics
- Benchmark additional small language models
- Package the translation model as a web or mobile application

---

## 📌 Notes

This is a research project focused on low-resource dialect translation, built under limited computational resources (single Colab T4 GPU). Results may vary depending on hardware, dataset version, and training hyperparameters.

---

## 🙏 Acknowledgements

Special thanks to my thesis supervisor **Ferdous Ara** for guidance throughout this project, as well as everyone who contributed sentence pairs, feedback, and support along the way.

---

## 📄 License

This project is released under the MIT License — see [`LICENSE`](LICENSE) for details.

---

## 📬 Contact

<div align="center">

**Sihabul Islam Safin**
Final-Year CSE Student, BGC Trust University Bangladesh · Freelance Generative AI Engineer

[![GitHub](https://img.shields.io/badge/GitHub-sihabsafin-181717?style=flat-square&logo=github)](https://github.com/sihabsafin/bangla-chittagonian-translation)

</div>
