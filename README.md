# 💬 Semantic Analysis on Twitter User Data using Hybrid Embedding Models  
*A Traditional Machine Learning Approach for Sentiment Classification*

---

## 🔍 Overview

This project explores the application of **Hybrid Embedding Models**—a synthesis of **TF-IDF** and **Word2Vec**—to perform sentiment classification on tweets. Developed as part of Coventry University's 7120CEM Natural Language Processing coursework, it classifies Twitter posts into *Positive*, *Neutral*, and *Negative* sentiment categories.

By integrating statistical and contextual textual representations, we aim to enhance classification accuracy using **Traditional Machine Learning Models** such as **Logistic Regression**, **SVM**, **Random Forest**, and **XGBoost**. The dataset is drawn from the **SemEval-2017 Task 4A**, comprising over 20,000 labeled tweets. Text preprocessing, hybrid feature engineering, model training, and evaluation are systematically executed.

The final models have relevance in real-world applications such as **brand monitoring**, **public opinion mining**, and **social behavior analytics**—demonstrating the practical value of academic NLP.

---

## 🧠 Project Objectives

- Build Hybrid Text Representations using TF-IDF and Word2Vec.
- Compare Classical Machine Learning Models for sentiment classification.
- Apply comprehensive preprocessing for social media text.
- Evaluate model performance with accuracy, F1-score, and ROC-AUC.
- Visualize insights via Word Clouds, Confusion Matrices, and ROC Curves.

---

## 🛠️ Technologies Used

- **Language:** Python 3.10+
- **Text Processing:** NLTK, re, string
- **Embeddings:** Gensim (Word2Vec), Scikit-learn (TF-IDF)
- **Models:** Logistic Regression, SVM, Random Forest, XGBoost
- **Evaluation:** Confusion Matrix, ROC Curve, Classification Report
- **Visualization:** Matplotlib, Seaborn, Wordcloud
- **IDE/Environment:** Google Colab (Primary), Jupyter

---

## 📁 Dataset

- **Source:** [SemEval-2017 Task 4A](https://alt.qcri.org/semeval2017/task4/)
- **Size:** 20,632 tweets
- **Classes:** Positive, Neutral, Negative
- **Preprocessing:** Deduplication, emoji & mention removal, tokenization, lemmatization

---

## 🔬 Methodology

1. **Data Cleaning & Preprocessing**
2. **Tokenization & Lemmatization**
3. **Feature Engineering**:
   - TF-IDF: Statistical frequency vectorization
   - Word2Vec: Semantic vector embedding (CBOW)
   - Hybrid: Concatenation of both vectors
4. **Model Implementation**:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - XGBoost
5. **Evaluation**:
   - Accuracy, F1-score, ROC-AUC
   - Confusion Matrix, ROC Curve

---

## 📊 Key Results

| Model             | TF-IDF | Word2Vec | Hybrid |
|------------------|--------|----------|--------|
| Logistic Regression | 82% | 74% | 82% |
| Random Forest      | 81% | 73% | 74% |
| SVM                | 83% | 74% | 79% |
| XGBoost            | 80% | 73% | 79% |

---

## 🧩 Real-World Applications

- **Political Sentiment Mining**
- **Brand Sentiment Tracking**
- **Crisis Alert Systems**
- **Market Research**
- **Online Reputation Management**

---

## 🔧 Environment and Setup Instructions

To set up the project locally:

```bash
git clone https://github.com/JaminUbuntu/NLP-Coursework-Benjamin.git
cd NLP-Coursework-Benjamin
pip install -r requirements.txt
```

---

## 📂 Folder Structure

```
NLP-Coursework-Benjamin/
├── data/                 # Cleaned dataset
├── models/               # Trained models (joblib)
├── outputs/              # Visualizations (ROC, Confusion Matrices)
├── utils/                # Preprocessing scripts
├── notebooks/            # Jupyter notebooks
├── requirements.txt
└── README.md
```

---

## 🧾 Licensing

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)  ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  

![Platform: Google Colab](https://img.shields.io/badge/platform-Colab-green.svg)  
![Status: Academic](https://img.shields.io/badge/status-submitted-blue)

---

## 🔍 Model Interpretability Tools

- Confusion Matrix
- ROC-AUC Curve
- Word Importance (TF-IDF)
- Feature Influence via SHAP (planned)

---

## 🖼️ Sample Output & Screenshots

| Word Cloud (Cleaned) | ROC Curve (Hybrid) |
|----------------------|--------------------|
| ![Word Cloud](outputs/wordcloud_cleaned.png) | ![ROC Curve](outputs/roc_curve_hybrid.png) |

---

## 💾 Model Inference

To reuse trained models:

```python
from joblib import load
model = load("models/svm_hybrid.joblib")
prediction = model.predict(new_data)
```

---

## 🧪 Contribution Guidelines

1. Fork this repository.
2. Create a branch: `git checkout -b feature-branch`
3. Commit your work: `git commit -m 'Add something cool'`
4. Push and create a Pull Request.

Adhere to [PEP8](https://peps.python.org/pep-0008/) and document all modules.

---

## 🎓 Academic Context

This repository supports the CW1 submission for **Module 7120CEM – Natural Language Processing** at **Coventry University**. It reflects core NLP competencies including:
- Linguistic Processing
- Machine Learning Implementation
- Dataset Design & Evaluation
- Ethical Research Practice

---

## 🏅 Badges

![Colab Ready](https://img.shields.io/badge/Notebook-Colab%20Compatible-brightgreen)
![TF-IDF + Word2Vec](https://img.shields.io/badge/Hybrid-TF--IDF%2BWord2Vec-blue)
![SemEval-2017](https://img.shields.io/badge/Dataset-SemEval2017-yellow)

---

## ❓ FAQ / Known Issues

- **Q: Why does Word2Vec underperform on its own?**  
  A: It requires larger training corpora to capture rich semantics effectively.

- **Q: Can this scale to multilingual tweets?**  
  A: Yes, with multilingual Word2Vec models and Unicode-aware preprocessing.

- **Q: Why use Traditional ML and not Deep Learning?**  
  A: To explore interpretable and lightweight models as a baseline.

---
