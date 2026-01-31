# Task 2: Credit Card Fraud Detection

## Objective
Develop a machine learning model to detect whether a transaction is **fraudulent** or **legitimate**.

This is a highly **imbalanced binary classification** problem.

---

## Dataset
Source: Kaggle Credit Card Fraud Detection Dataset  
Dataset link is provided in `dataset_link.txt`.

Files:
- `fraudTrain.csv`
- `fraudTest.csv`

---

## Approach
1. Load dataset using optimized CSV reading (sampling/chunking for speed)
2. Feature engineering:
   - Date-time feature extraction (year/month/day/hour/dayofweek)
3. Preprocessing:
   - Label encoding categorical variables
   - Standard scaling numeric variables
4. Model training:
   - Logistic Regression with `class_weight='balanced'`
5. Evaluation:
   - ROC-AUC
   - PR-AUC
   - Precision / Recall / F1-score for fraud class
   - Confusion matrix

---

## Results
Example validation performance:
- ROC-AUC: ~0.87
- PR-AUC: ~0.26
- Recall (fraud): ~0.77

---

## How to Run
```bash
pip install -r ../requirements.txt
python src/fraud_detection.py
```

Author
Sravan
CODSOFT ML Intern
