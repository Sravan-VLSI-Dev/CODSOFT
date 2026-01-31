# Task 1: Movie Genre Classification

## Objective
Build a machine learning model to **predict the genre of a movie** using its **plot summary/description** (NLP classification).

---

## Dataset
Source: Kaggle Movie Genre Classification Dataset  
Dataset link is provided in `dataset_link.txt`.

Dataset file format uses delimiter `:::`.

- Train / Solution format:
ID ::: TITLE ::: GENRE ::: DESCRIPTION

- Test format:
ID ::: TITLE ::: DESCRIPTION


---

## Approach
1. Parse `.txt` dataset using delimiter `:::`
2. Text preprocessing and cleaning
3. Feature extraction using **TF-IDF Vectorizer**
 - `ngram_range=(1,2)`
 - `max_features=80000`
 - `sublinear_tf=True`
4. Classification using **LinearSVC (Linear SVM)**
5. Evaluation on test set using `test_data_solution.txt`

---

## Model
- Vectorizer: TF-IDF (unigram + bigram)
- Classifier: LinearSVC
- Metric: Accuracy + classification report

---

## Results
Achieved accuracy around **60%** on test dataset (improved from baseline using n-grams and feature tuning).

---

## How to Run
```bash
pip install -r ../requirements.txt
python src/movie_genre_final.py
```

Author
Sravan
CODSOFT ML Intern
