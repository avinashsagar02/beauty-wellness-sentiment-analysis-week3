# Beauty & Wellness Customer Sentiment & Feedback Analysis

[![Internship Track](https://img.shields.io/badge/Internship-Data%20Science%20with%20Python-blue.svg)](https://github.com)
[![Deliverable](https://img.shields.io/badge/Deliverable-Week%203%20Task-green.svg)](https://github.com)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)

## 📌 Project Overview
This repository contains the end-to-end data processing, exploratory data analysis, and natural language preprocessing architecture for evaluating consumer sentiment, product efficacy, and feedback patterns in the **Beauty and Personal Care E-Commerce sector**.

While aggregate star ratings (1 to 5 stars) provide broad indicators of customer satisfaction, they fail to isolate underlying defect drivers—such as chemical allergies, formulation breakouts, and dispensing pump malfunctions. This project implements an automated, reproducible Natural Language Processing (NLP) framework to extract actionable intelligence from unstructured consumer reviews.

---

## 🎯 Week 3 Objectives: Data Cleaning & Preprocessing Pipeline
- **5-Gate Pipeline Architecture:** Designing an end-to-end modular pipeline covering raw ingestion, missing value resolution, outlier detection, regex text normalization, and feature scaling.
- **Innovative Data Quality Remediation:** 
  - Preserving demographic volume by imputing missing skin attributes with `'Not Specified'` tokens rather than discarding valuable consumer text.
  - Mitigating negation loss by engineering custom domain-aware stopword dictionaries that retain essential polarity qualifiers (`not`, `never`, `hardly`, `without`).
  - IQR-based outlier boundaries eliminating low-information bot spam and micro-reviews.
- **Audit of Data Biases & Limitations:** Comprehensive analysis of positivity skew (72% 5-star bias), demographic self-selection bias, and promotional PR sample noise.
- **Reproducibility Standards:** Enforcing deterministic seed parameters, automated null-frequency tracking, and logging execution checksums.

---

## 📂 Repository Structure
```text
├── README.md                    # Project documentation & preprocessing architecture
├── data_cleaning_pipeline.py    # Week 3: Production 5-gate cleaning & preprocessing pipeline
├── eda_script.py                # Week 2: Initial exploratory data analysis & N-gram modeling
└── requirements.txt             # Project environment dependencies
```

---

## 🏗️ 5-Gate Preprocessing Architecture Flowchart

```text
+-----------------------------------------------------------------------------------------+
|                       GATE 1: RAW INGESTION & DATA INTEGRITY AUDIT                     |
|   - Ingest CSV/JSON corpuses (Sephora & Amazon Beauty) via Pandas.                     |
|   - Audit structural types, record counts, null rates, and duplicate review IDs.       |
+-------------------------------------------+---------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
|                     GATE 2: MISSING VALUE & SPARSITY RESOLUTION                         |
|   - Drop unrecoverable nulls (Review_Text, Star_Rating).                               |
|   - Categorical imputation: Impute optional demographic attributes with 'Not Specified'.|
|   - Numerical median imputation for continuous attributes (Price, Helpfulness).        |
+-------------------------------------------+---------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
|                    GATE 3: OUTLIER DETECTION & IRREGULARITY FILTERING                   |
|   - Statistical IQR filtering on review word lengths (cut off bot spam & micro-reviews).|
|   - Inconsistent rating clipping (constrain strictly to 1.0 - 5.0 range).               |
|   - Non-English character and language-detection filtering.                            |
+-------------------------------------------+---------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
|                     GATE 4: TEXT SANITIZATION & NLP NORMALIZATION                       |
|   - Regex cleaning: Strip HTML tags, special symbols, multiple spaces, URLs.            |
|   - Lowercasing & Contraction expansion ('can't' -> 'cannot', 'didn't' -> 'did not').   |
|   - Sentiment-preserving stopword removal & lemmatization via NLTK/spaCy.               |
+-------------------------------------------+---------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
|                   GATE 5: DATA NORMALIZATION, ENCODING & EXPORT                         |
|   - Standardize dates to ISO-8601 (YYYY-MM-DD); MinMax scale numerical features.       |
|   - One-hot encode product categories; export clean parquet/csv with audit checksum.   |
+-----------------------------------------------------------------------------------------+
```

---

## 🚀 Execution Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/avinashsagar02/beauty-wellness-sentiment-analysis.git
cd beauty-wellness-sentiment-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Preprocessing Pipeline
```bash
python data_cleaning_pipeline.py
```

---

## 🛠️ Tech Stack & Tooling Justification
- **Python 3.10+**: Core programming language runtime.
- **Pandas**: Vectorized operations, missing data handling, and schema transformations.
- **NumPy**: Statistical IQR outlier detection and array calculations.
- **Regular Expressions (`re`)**: Ultra-fast pattern matching for stripping HTML artifacts and noisy non-ASCII symbols.
- **Scikit-Learn (`MinMaxScaler`)**: Normalizing numerical features for downstream modeling convergence.
