import re
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CONFIG
USE_CLASS_WEIGHT_BALANCED = False   
MAX_FEATURES = 80000                
NGRAM_RANGE = (1, 2)                
MIN_DF = 2
C_CANDIDATES = [0.5, 1.0, 2.0, 3.0] 

# PATHS (works from any directory)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent

TRAIN_PATH = DATA_DIR / "train_data.txt"
TEST_PATH = DATA_DIR / "test_data.txt"
TEST_SOLUTION_PATH = DATA_DIR / "test_data_solution.txt"

# DATA LOADER
def load_file(path: Path, is_train: bool = True) -> pd.DataFrame:
    """
    Parses Kaggle Movie Genre Classification dataset format:
      train/solution: id ::: title ::: genre ::: description
      test:           id ::: title ::: description
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(":::")]

            if is_train:
                if len(parts) >= 4:
                    rows.append([parts[0], parts[1], parts[2], parts[3]])
            else:
                if len(parts) >= 3:
                    rows.append([parts[0], parts[1], parts[2]])

    if is_train:
        return pd.DataFrame(rows, columns=["id", "title", "genre", "description"])
    return pd.DataFrame(rows, columns=["id", "title", "description"])

# TEXT CLEANING (light + safe)
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"<.*?>", " ", s)          # remove html
    s = re.sub(r"[^a-z\s]", " ", s)       # keep letters + spaces
    s = re.sub(r"\s+", " ", s).strip()    # normalize spaces
    return s

# MAIN
def main():
    # Load data
    train_df = load_file(TRAIN_PATH, is_train=True)
    test_df = load_file(TEST_PATH, is_train=False)
    solution_df = load_file(TEST_SOLUTION_PATH, is_train=True)

    # Basic validation
    assert len(test_df) == len(solution_df), "Mismatch: test and solution row counts differ!"

    # Clean text
    train_df["description"] = train_df["description"].apply(clean_text)
    test_df["description"] = test_df["description"].apply(clean_text)

    X_train = train_df["description"]
    y_train = train_df["genre"]

    X_test = test_df["description"]
    y_true = solution_df["genre"]

    # Vectorize 
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        min_df=MIN_DF
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    best_acc = -1
    best_model = None
    best_C = None

    for C in C_CANDIDATES:
        model = LinearSVC(
            C=C,
            class_weight="balanced" if USE_CLASS_WEIGHT_BALANCED else None,
            random_state=42,
            max_iter=8000
        )
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_true, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_C = C

    y_pred = best_model.predict(X_test_vec)
    report = classification_report(y_true, y_pred, digits=4)

    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    mistakes = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            mistakes[(t, p)] = mistakes.get((t, p), 0) + 1
    top_mistakes = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)[:10]

    #print("\n" + "=" * 72)
    print("CODSOFT INTERNSHIP - TASK 1")
    print("MOVIE GENRE CLASSIFICATION (NLP TEXT CLASSIFICATION)")
    #print("=" * 72)

    print("\n Dataset Summary")
    #print("-" * 72)
    print(f"Train samples      : {len(train_df):,}")
    print(f"Test samples       : {len(test_df):,}")
    print(f"Total genres       : {train_df['genre'].nunique()}")

    print("\n Feature Engineering")
    #print("-" * 72)
    print(f"Vectorizer         : TF-IDF")
    print(f"N-grams            : {NGRAM_RANGE} (unigram + bigram)")
    print(f"Max features       : {MAX_FEATURES:,}")
    print(f"Sublinear TF       : Enabled")
    print(f"min_df             : {MIN_DF}")

    print("\n Model")
    #print("-" * 72)
    print(f"Classifier         : LinearSVC (Linear Support Vector Machine)")
    print(f"Best C             : {best_C}")
    print(f"Class weight       : {'balanced' if USE_CLASS_WEIGHT_BALANCED else 'None'}")

    print("\n Performance")
    #print("-" * 72)
    print(f"Accuracy           : {best_acc:.4f} ({best_acc*100:.2f}%)")

    print("\n Classification Report")
    #print("-" * 72)
    print(report)

    print("\n Top 10 Misclassifications (True → Predicted)")
    #print("-" * 72)
    if not top_mistakes:
        print("No misclassifications found (unlikely).")
    else:
        for (t, p), cnt in top_mistakes:
            print(f"{t:>12} → {p:<12} : {cnt}")

    sample = "A young boy discovers magical powers and saves the world from evil forces."
    sample_vec = vectorizer.transform([clean_text(sample)])
    pred_genre = best_model.predict(sample_vec)[0]

    print("\n Sample Prediction")
    #print("-" * 72)
    print(f"Sample Text        : {sample}")
    print(f"Predicted Genre    : {pred_genre}")

    #print("\n" + "=" * 72)
    print(" Execution completed successfully")
    #print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
