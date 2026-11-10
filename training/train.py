"""Trains the Toxic Lens model and exports everything the browser needs.

Usage: python training/train.py --data <folder with the two CSVs> --out <repo root>

The dataset is not stored in this repository. Download re_dataset.csv and
new_kamusalay.csv from
https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split

SEED = 42
GRID = {"C": [0.5, 1, 2, 4, 8], "ngram_max": [1, 2], "max_features": [10000, 20000]}
FIXTURES = [
    "Dasar BODOH, kamu tidak berguna!!!",
    "Terima kasih banyak atas bantuannya ya kak",
    "gue gak suka sama lo, bangeeet https://contoh.com @teman",
    "Semoga harimu menyenangkan dan penuh berkah",
    "RT @user pemerintah harus lebih peduli pada rakyat kecil",
    "",
]


def clean(text):
    t = text.lower()
    t = re.sub(r"\\x[0-9a-f]{2}", " ", t)  # the dataset stores emoji as escaped bytes
    t = re.sub(r"\\n", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\b(rt|user|url)\b", " ", t)
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize(cleaned, slang):
    out = []
    for token in cleaned.split():
        out.extend(slang.get(token, token).split())
    return out


def load_slang(path):
    slang = {}
    with open(path, encoding="latin-1", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            key, value = clean(row[0]), clean(row[1])
            if key and value and " " not in key and key != value:
                slang[key] = value
    return slang


def vectorizer(params):
    return TfidfVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False, preprocessor=None,
        ngram_range=(1, params["ngram_max"]), max_features=params["max_features"],
        min_df=2, sublinear_tf=True, norm="l2", smooth_idf=True,
    )


def fit(params, texts, labels):
    vec = vectorizer(params)
    model = LogisticRegression(C=params["C"], max_iter=3000, random_state=SEED)
    model.fit(vec.fit_transform(texts), labels)
    return vec, model


def scores(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "rocAuc": round(roc_auc_score(y_true, y_prob), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def score_text(text, slang, vec, model):
    tokens = normalize(clean(text), slang)
    row = vec.transform([" ".join(tokens)])
    logit = float(model.decision_function(row)[0])
    return {"text": text, "tokens": tokens, "logit": round(logit, 4), "probability": round(1 / (1 + math.exp(-logit)), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    slang = load_slang(args.data / "new_kamusalay.csv")
    raw = pd.read_csv(args.data / "re_dataset.csv", encoding="latin-1")
    frame = pd.DataFrame({
        "text": [" ".join(normalize(clean(t), slang)) for t in raw["Tweet"]],
        "label": ((raw["HS"] == 1) | (raw["Abusive"] == 1)).astype(int),
    })
    frame = frame[frame["text"] != ""]
    before = len(frame)
    # Deduplicating before the split stops identical tweets leaking into the test set.
    frame = frame.groupby("text", as_index=False)["label"].max()
    print(f"rows {len(raw)} -> {before} non-empty -> {len(frame)} unique, slang entries {len(slang)}")

    train, rest = train_test_split(frame, test_size=0.3, random_state=SEED, stratify=frame["label"])
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=rest["label"])

    best, best_f1 = None, -1.0
    for params in ParameterGrid(GRID):
        vec, model = fit(params, train["text"], train["label"])
        val_f1 = f1_score(val["label"], model.predict(vec.transform(val["text"])))
        print(params, round(val_f1, 4))
        if val_f1 > best_f1:
            best, best_f1 = params, val_f1
    print("best", best, round(best_f1, 4))

    development = pd.concat([train, val])
    vec, model = fit(best, development["text"], development["label"])
    test_metrics = scores(test["label"].to_numpy(), model.predict_proba(vec.transform(test["text"]))[:, 1])
    majority = max(test["label"].mean(), 1 - test["label"].mean())
    print("test", test_metrics, "majority baseline", round(majority, 4))

    idf = vec.idf_
    coef = model.coef_[0]
    vocabulary = {term: [round(float(idf[i]), 5), round(float(coef[i]), 5)] for term, i in vec.vocabulary_.items()}

    (args.out / "model").mkdir(parents=True, exist_ok=True)
    (args.out / "test").mkdir(parents=True, exist_ok=True)
    payload = {"intercept": round(float(model.intercept_[0]), 5), "ngramMax": best["ngram_max"], "vocabulary": vocabulary, "slang": slang}
    (args.out / "model" / "model.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    metrics = {
        "algorithm": "TF-IDF + Logistic Regression",
        "params": best,
        "sizes": {"source": len(raw), "unique": len(frame), "development": len(development), "test": len(test), "features": len(vocabulary), "slang": len(slang)},
        "toxicShare": round(float(frame["label"].mean()), 4),
        "majorityBaseline": round(float(majority), 4),
        "test": test_metrics,
    }
    (args.out / "model" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fixtures = [score_text(t, slang, vec, model) for t in FIXTURES]
    (args.out / "test" / "fixtures.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")
    print("model bytes", (args.out / "model" / "model.json").stat().st_size)
    for f in fixtures:
        print(f["probability"], f["tokens"])


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
