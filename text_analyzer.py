"""
TASK 1: Text Analyzer (Fake News Detection)
Method: TF-IDF + Logistic Regression (Scikit-learn)

Dataset (free, Kaggle): "Fake and Real News Dataset"
Download Fake.csv and True.csv and put them in the data/ folder.
Then run:  python text_analyzer.py   (trains and saves the model)
"""
import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

MODEL_PATH = "text_model.joblib"
FAKE_CSV = os.path.join("data", "Fake.csv")
TRUE_CSV = os.path.join("data", "True.csv")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\(reuters\)|reuters", " ", text)  # avoid dataset leakage
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7, max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_model():
    """Train on the Kaggle dataset. Returns test accuracy."""
    fake = pd.read_csv(FAKE_CSV)
    true = pd.read_csv(TRUE_CSV)
    fake["label"] = 0   # 0 = Fake
    true["label"] = 1   # 1 = Real
    df = pd.concat([fake, true], ignore_index=True)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    model = build_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Test accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, pred, target_names=["Fake", "Real"]))
    joblib.dump(model, MODEL_PATH)
    return acc


def _demo_model():
    """Tiny fallback so the app runs even without the dataset.
    NOT accurate - only for UI demo. Train on the real dataset for real results."""
    texts = [
        "government announces new budget for education and health sector",
        "city council approves funding for new public road project",
        "scientists publish study on climate change in peer reviewed journal",
        "central bank keeps interest rates unchanged after policy meeting",
        "shocking secret miracle cure doctors do not want you to know",
        "you won't believe what this celebrity did, share before deleted",
        "aliens secretly control world government leaked insider reveals",
        "breaking!!! drink this to cure all diseases overnight, 100% proven",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    m = build_pipeline()
    m.fit([clean_text(t) for t in texts], labels)
    return m


_model = None
_is_demo = False


def load_model():
    global _model, _is_demo
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        elif os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV):
            train_model()
            _model = joblib.load(MODEL_PATH)
        else:
            _model, _is_demo = _demo_model(), True
    return _model


def predict_text(text):
    """Returns dict: label ('Real'/'Fake'), confidence (%), demo_mode (bool)."""
    model = load_model()
    proba = model.predict_proba([clean_text(text)])[0]   # [P(fake), P(real)]
    idx = int(proba.argmax())
    return {
        "label": "Real" if idx == 1 else "Fake",
        "confidence": round(float(proba[idx]) * 100, 2),
        "demo_mode": _is_demo,
    }


if __name__ == "__main__":
    train_model()
