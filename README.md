# CODSOFT - Machine Learning Internship Tasks

This repository contains the tasks completed as part of my **Machine Learning Internship at CODSOFT**.

Each task is organized as an independent project with:
- Clean folder structure (`src/`, `data/`, `outputs/`)
- Reproducible execution steps
- Professional documentation
- Model evaluation and outputs

---

## Repository Structure

CODSOFT/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Task_1_Movie_Genre_Classification/
├── Task_2_Credit_Card_Fraud_Detection/
└── Task_3_Customer_Churn_Prediction/


---

## Tasks Completed

###  Task 1: Movie Genre Classification
- Type: NLP multi-class classification
- Model: TF-IDF (unigrams + bigrams) + Linear SVM (LinearSVC)
- Output: Predicted genre from plot summary

 Folder: `Task_1_Movie_Genre_Classification/`

---

###  Task 2: Credit Card Fraud Detection
- Type: Binary classification (highly imbalanced)
- Model: Logistic Regression with `class_weight='balanced'`
- Evaluation: ROC-AUC, PR-AUC, fraud recall/precision

 Folder: `Task_2_Credit_Card_Fraud_Detection/`

---

###  Task 3: Customer Churn Prediction
- Type: Binary classification
- Model: Random Forest (selected using ROC-AUC)
- Interpretability: Feature importance ranking

 Folder: `Task_3_Customer_Churn_Prediction/`

---

## Setup Instructions

### 1) Clone Repository
```bash
git clone https://github.com/<your-username>/CODSOFT.git
cd CODSOFT
2) Create Virtual Environment (Recommended)
python -m venv .venv
Activate:

Windows PowerShell

.venv\Scripts\Activate
3) Install Dependencies
pip install -r requirements.txt
Run Tasks
Task 1
python Task_1_Movie_Genre_Classification/src/movie_genre_final.py
Task 2
python Task_2_Credit_Card_Fraud_Detection/src/fraud_detection.py
Task 3
python Task_3_Customer_Churn_Prediction/src/churn_prediction.py 
```

Author
Sravan
CODSOFT Machine Learning Intern
