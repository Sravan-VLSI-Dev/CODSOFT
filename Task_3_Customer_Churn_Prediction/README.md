# Task 3: Customer Churn Prediction

## Objective
Predict whether a customer will **churn (Exited = 1)** or **stay retained (Exited = 0)**.

---

## Dataset
Source: Kaggle - Bank Customer Churn Modelling Dataset  
Dataset link is provided in `dataset_link.txt`.

File used:
- `Churn_Modelling.csv`

Target column:
- `Exited`

---

## Approach
1. Load dataset and drop identifier columns (`RowNumber`, `CustomerId`, `Surname`)
2. Handle missing values:
   - median for numeric
   - mode for categorical
3. Preprocessing pipeline:
   - OneHotEncoder for categorical columns (`Geography`, `Gender`)
   - StandardScaler for numeric columns
4. Train and select best model using ROC-AUC:
   - Logistic Regression
   - Random Forest
5. Evaluate using:
   - Accuracy
   - ROC-AUC
   - Confusion matrix
   - Classification report
6. Interpretability:
   - Feature importance ranking from Random Forest

---

## Results
Example validation performance:
- Accuracy: ~85%
- ROC-AUC: ~0.86

---

## How to Run
```bash
pip install -r ../requirements.txt
python src/churn_prediction.py
```

Author
Sravan
CODSOFT ML Intern
