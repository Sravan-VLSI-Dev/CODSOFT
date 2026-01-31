import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# --------- File Paths using pathlib (works from any directory) ---------
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Data files are in the parent directory (Task1_MovieGenreClassification/data/)
DATA_DIR = SCRIPT_DIR.parent

TRAIN_PATH = DATA_DIR / "train_data.txt"
TEST_PATH = DATA_DIR / "test_data.txt"
TEST_SOLUTION_PATH = DATA_DIR / "test_data_solution.txt"

print("="*70)
print("🔍 FILE PATH VERIFICATION")
print("="*70)
print(f"Script location: {SCRIPT_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"\nResolved file paths:")
print(f"  Train file:    {TRAIN_PATH}")
print(f"  Test file:     {TEST_PATH}")
print(f"  Solution file: {TEST_SOLUTION_PATH}")

# --------- Verify all files exist ----------
print(f"\n📋 FILE EXISTENCE CHECK:")
files_to_check = {
    "train_data.txt": TRAIN_PATH,
    "test_data.txt": TEST_PATH,
    "test_data_solution.txt": TEST_SOLUTION_PATH
}

all_files_exist = True
for file_name, file_path in files_to_check.items():
    exists = file_path.exists()
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"  {file_name}: {status}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    raise FileNotFoundError("❌ One or more data files are missing!")

# --------- Helper function to load files ----------
def load_file(path, is_train=True):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(":::")
            parts = [p.strip() for p in parts]

            if is_train:
                # Expected: id ::: title ::: genre ::: description
                if len(parts) >= 4:
                    rows.append([parts[0], parts[1], parts[2], parts[3]])
            else:
                # Expected: id ::: title ::: description
                if len(parts) >= 3:
                    rows.append([parts[0], parts[1], parts[2]])

    if is_train:
        return pd.DataFrame(rows, columns=["id", "title", "genre", "description"])
    else:
        return pd.DataFrame(rows, columns=["id", "title", "description"])


# --------- Load datasets ----------
print(f"\n📂 LOADING DATASETS")
train_df = load_file(TRAIN_PATH, is_train=True)
test_df = load_file(TEST_PATH, is_train=False)
solution_df = load_file(TEST_SOLUTION_PATH, is_train=True)

# --------- Debug: Print first 2 lines from each file ----------
print(f"\n🔎 DELIMITER PARSING VERIFICATION (first 2 lines):")
print(f"\n📄 Train data (first 2 rows):")
print(train_df.head(2).to_string())

print(f"\n📄 Test data (first 2 rows):")
print(test_df.head(2).to_string())

print(f"\n📄 Solution data (first 2 rows):")
print(solution_df.head(2).to_string())

# --------- Verify dataframe shapes and columns ----------
print(f"\n📊 DATAFRAME SHAPES AND COLUMNS:")
print(f"\n✓ Train DataFrame:")
print(f"  Shape: {train_df.shape}")
print(f"  Columns: {list(train_df.columns)}")
print(f"  Expected: ['id', 'title', 'genre', 'description']")
assert list(train_df.columns) == ["id", "title", "genre", "description"], "Train columns mismatch!"

print(f"\n✓ Test DataFrame:")
print(f"  Shape: {test_df.shape}")
print(f"  Columns: {list(test_df.columns)}")
print(f"  Expected: ['id', 'title', 'description']")
assert list(test_df.columns) == ["id", "title", "description"], "Test columns mismatch!"

print(f"\n✓ Solution DataFrame:")
print(f"  Shape: {solution_df.shape}")
print(f"  Columns: {list(solution_df.columns)}")
print(f"  Expected: ['id', 'title', 'genre', 'description']")
assert list(solution_df.columns) == ["id", "title", "genre", "description"], "Solution columns mismatch!"

# --------- Train Model ----------
print(f"\n🎓 MODEL TRAINING")
print(f"Features: TF-IDF Vectorizer (stop_words='english', max_features=30000)")
print(f"Algorithm: LinearSVC")

X_train = train_df["description"]
y_train = train_df["genre"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
X_train_vec = vectorizer.fit_transform(X_train)

print(f"Training samples: {X_train_vec.shape[0]}")
print(f"Feature dimension: {X_train_vec.shape[1]}")

model = LinearSVC(random_state=42, max_iter=1000)
model.fit(X_train_vec, y_train)
print(f"✅ Model training completed!")

# --------- Evaluate on Test Data (using solution file) ----------
print(f"\n📈 MODEL EVALUATION")
X_test_vec = vectorizer.transform(test_df["description"])
y_pred = model.predict(X_test_vec)

y_true = solution_df["genre"]

acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Model Accuracy on Test Set: {acc:.4f} ({acc*100:.2f}%)")

print(f"\n📋 Classification Report:")
print(classification_report(y_true, y_pred))

# --------- Try custom prediction ----------
print(f"\n🧪 SAMPLE PREDICTION TEST:")
sample = ["A young boy discovers magical powers and saves the world from evil forces."]
sample_vec = vectorizer.transform(sample)
predicted_genre = model.predict(sample_vec)[0]
print(f"Sample text: '{sample[0]}'")
print(f"Predicted genre: {predicted_genre}")

print("\n" + "="*70)
print("✅ Script execution completed successfully!")
print("="*70)
