import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# --------- File Paths using pathlib (works from any directory) ---------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent  # data folder

TRAIN_PATH = DATA_DIR / "train_data.txt"
TEST_PATH = DATA_DIR / "test_data.txt"
TEST_SOLUTION_PATH = DATA_DIR / "test_data_solution.txt"


# --------- Helper function to load dataset files ----------
def load_file(path, is_train=True):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(":::")]

            if is_train:
                # train/solution format: id ::: title ::: genre ::: description
                if len(parts) >= 4:
                    rows.append([parts[0], parts[1], parts[2], parts[3]])
            else:
                # test format: id ::: title ::: description
                if len(parts) >= 3:
                    rows.append([parts[0], parts[1], parts[2]])

    if is_train:
        return pd.DataFrame(rows, columns=["id", "title", "genre", "description"])
    else:
        return pd.DataFrame(rows, columns=["id", "title", "description"])


# --------- Load datasets ----------
train_df = load_file(TRAIN_PATH, is_train=True)
test_df = load_file(TEST_PATH, is_train=False)
solution_df = load_file(TEST_SOLUTION_PATH, is_train=True)


# --------- Train Model ----------
X_train = train_df["description"]
y_train = train_df["genre"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LinearSVC(random_state=42, max_iter=2000)
model.fit(X_train_vec, y_train)


# --------- Evaluate on Test Data ----------
X_test_vec = vectorizer.transform(test_df["description"])
y_pred = model.predict(X_test_vec)
y_true = solution_df["genre"]

acc = accuracy_score(y_true, y_pred)

print("=" * 60)
print("MOVIE GENRE CLASSIFICATION - RESULTS")
print("=" * 60)
print(f"Training samples : {len(train_df)}")
print(f"Testing samples  : {len(test_df)}")
print(f"TF-IDF features  : {X_train_vec.shape[1]}")
print(f"Model            : LinearSVC")
print(f"Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
print("=" * 60)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))


# --------- Sample prediction ----------
sample = "A young boy discovers magical powers and saves the world from evil forces."
sample_vec = vectorizer.transform([sample])
predicted_genre = model.predict(sample_vec)[0]

print("\nSample Prediction:")
print("Text   :", sample)
print("Genre  :", predicted_genre)
