# Text Mining — Feature Selection for LSTM

> Comparative study of three feature selection approaches applied to LSTM-based text classification on two distinct datasets: Arabic sentiment analysis and English medical text classification.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Overview

This project investigates whether feature selection methods can improve the performance of LSTM models for text classification while reducing the vocabulary size. Three approaches are compared:

1. **Baseline LSTM** — No feature selection (10 000 TF-IDF features)
2. **Mutual Information (MI)** — Statistical scoring of word relevance
3. **MI + Genetic Algorithm (MI+GA)** — Hybrid approach combining MI filtering and evolutionary optimization

The methods are evaluated on two datasets in different languages and domains to assess their generalization.

---

## 📊 Datasets

| Dataset | Language | Domain | Size | Classes | Avg. Length |
|---------|----------|--------|------|---------|-------------|
| **DS1** Medical Text | English | Medical research | 14 438 docs | 5 | 179 tokens |
| **DS2** Arabic Sentiment | Arabic | Twitter sentiment | 8 622 tweets | 2 | 24 tokens |

- **DS1 Source:** [Kaggle — chaitanyakck/medical-text](https://www.kaggle.com/datasets/chaitanyakck/medical-text)
- **DS2 Source:** [Kaggle — mksaad/arabic-sentiment-twitter-corpus](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus)

---

##  Methods

### 1. Baseline LSTM
Standard architecture with full TF-IDF vocabulary:
For DS1, an enhanced **BiLSTM + GlobalMaxPooling** architecture was also tested.

### 2. Mutual Information (MI)
Computes the statistical dependence between each word and the class label:
The top-k most informative words are kept (k tested: 300, 500, 700, 1000, 1500, 2000).

### 3. MI + Genetic Algorithm (MI+GA)
Two-stage hybrid approach:
- **Stage 1 (MI):** Reduce search space from 10 000 → top 1 000 features
- **Stage 2 (GA):** Find optimal feature subset using evolutionary search
  - Population: 10–30 individuals
  - Generations: 10–20
  - Crossover: Two-point (CXPB=0.7)
  - Mutation: Bit-flip (MUTPB=0.01)
  - Selection: Tournament (size=3)
  - Fitness: F1 macro of LSTM classifier on validation set

---

##  Results

### DS1 — Medical Text (English)

| Method | Features | Accuracy | F1 Macro |
|--------|----------|----------|----------|
| LSTM Baseline | 10 000 | 0.5531 | 0.5466 |
| **BiLSTM Baseline** ⭐ | **10 000** | **0.6122** | **0.5989** |
| MI k=2000 | 2 000 | 0.6122 | 0.5979 |
| MI k=1000 | 1 000 | 0.6034 | 0.5900 |
| MI+GA+BiLSTM | 507 | 0.5762 | 0.5489 |

**Key insight:** MI k=2000 matches the BiLSTM baseline accuracy with **80% fewer features**.

### DS2 — Arabic Sentiment (Twitter)

| Method | Features | Accuracy | F1 Macro |
|--------|----------|----------|----------|
| **LSTM Baseline** ⭐ | **10 000** | **0.6337** | **0.6240** |
| MI k=700 | 700 | 0.6167 | 0.6030 |
| MI k=1500 | 1 500 | 0.5989 | 0.5549 |
| MI k=300 | 300 | 0.5641 | 0.4998 |
| MI+GA+LSTM | 521 | 0.5703 | 0.5024 |

**Key insight:** MI k=700 offers the best compression/performance trade-off: **-93% features for only -1.7% accuracy**.

---

##  Key Findings

1. **Baseline resists feature selection** — On both datasets, no FS method outperforms the baseline
2. **MI is highly effective for compression** — On DS1, MI k=2000 matches baseline with 80% fewer features
3. **GA suffers from premature convergence** — Small population sizes lead to local optima
4. **Sequential models need contextual richness** — Reducing vocabulary too aggressively breaks LSTM context
5. **Short corpora amplify the problem** — Arabic tweets (24 tokens) suffer more than medical docs (179 tokens)

---

##  Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — LSTM and BiLSTM models
- **scikit-learn** — TF-IDF, Mutual Information, metrics
- **DEAP** — Genetic Algorithm framework
- **NLTK** — Arabic stopwords
- **arabic-reshaper + python-bidi** — Arabic visualization
- **pandas / numpy / matplotlib / seaborn** — Data processing and visualization

---

##  Project Structure
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── DS1_Medical_EN_Complete.ipynb
│   └── DS2_Arabic_Sentiment_Complete.ipynb
├── src/
│   └── models/
│       └── lstm_model.py
├── results/
│   ├── figures/
│   └── scores/
├── reports/
│   ├── presentation.pptx
│   └── rapport.pdf
├── requirements.txt
└── README.md

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/iliasNdm/text-mining-feature-selection-lstm.git
cd text-mining-feature-selection-lstm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Usage

```bash
# Run DS2 Arabic Sentiment notebook
jupyter notebook notebooks/DS2_Arabic_Sentiment_Complete.ipynb

# Run DS1 Medical Text notebook
jupyter notebook notebooks/DS1_Medical_EN_Complete.ipynb
```

For GA experiments, GPU is recommended (Google Colab T4 was used in this study).

---

##  Future Work

- **Pre-trained embeddings** — Integrate AraBERT (DS2) and BioBERT (DS1)
- **BiLSTM for DS2** — Apply bidirectional architecture to Arabic sentiment
- **CAMeL Tools** — Improve Arabic preprocessing (prefix separation)
- **Aligned fitness function** — Use the final classifier directly as GA fitness
- **Data augmentation** — Back-translation and paraphrasing to expand corpora
- **Larger GA populations** — Reduce premature convergence

---


##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---


