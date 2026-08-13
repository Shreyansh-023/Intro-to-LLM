import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm


LABELS = [
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "gom_Deva",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "kas_Arab",
    "khasi",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Mtei",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "san_Deva",
    "santhali",
    "snd_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
]

MULTISPACE = re.compile(r"\s+")


TF_MODES = {
    "raw": "Unnormalized TF (raw term count)",
    "len_norm": "TF normalized by sentence length (total words)",
    "max_norm": "TF normalized by most frequent word in sentence",
}

IDF_MODES = {
    "raw": "Unnormalized IDF",
    "maxword_norm": "IDF normalized using the most frequent word across sentences",
}


def normalize_text(text):
    return MULTISPACE.sub(" ", text.strip())


def tokenize(text):
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def read_rows(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row.get("text", "")
            label = row.get("label", "")
            if text and label:
                rows.append({"text": text, "label": label})
    return rows


class LabelEncoder:
    def __init__(self, labels):
        self.labels = list(labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}

    def encode(self, labels):
        return np.array([self.label_to_index[label] for label in labels], dtype=np.int64)

    def decode(self, indices):
        return [self.labels[index] for index in indices]


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list
    val_macro_f1: list


class LogisticRegressionOVR:
    def __init__(self, num_features, num_classes, learning_rate=0.5, epochs=20, l2=1e-4):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights = np.zeros((num_features, num_classes), dtype=np.float64)
        self.bias = np.zeros(num_classes, dtype=np.float64)

    @staticmethod
    def softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict_proba(self, features):
        logits = features @ self.weights + self.bias
        return self.softmax(np.asarray(logits))

    def predict(self, features):
        probabilities = self.predict_proba(features)
        return np.argmax(probabilities, axis=1)

    def compute_loss(self, features, labels):
        probabilities = self.predict_proba(features)
        clipped = np.clip(probabilities, 1e-12, 1.0)
        nll = -np.log(clipped[np.arange(len(labels)), labels]).mean()
        regularization = 0.5 * self.l2 * np.sum(self.weights * self.weights)
        return nll + regularization

    def fit(self, train_x, train_y, val_x=None, val_y=None):
        history = TrainingHistory(train_loss=[], val_loss=[], val_macro_f1=[])
        identity = np.eye(self.num_classes, dtype=np.float64)
        one_hot = identity[train_y]
        sample_count = train_x.shape[0]

        for epoch in range(1, self.epochs + 1):
            probabilities = self.predict_proba(train_x)
            errors = (probabilities - one_hot) / sample_count
            grad_w = train_x.T @ errors + self.l2 * self.weights
            grad_b = errors.sum(axis=0)

            self.weights -= self.learning_rate * np.asarray(grad_w)
            self.bias -= self.learning_rate * grad_b

            train_loss = self.compute_loss(train_x, train_y)
            history.train_loss.append(float(train_loss))

            message = f"Epoch {epoch:02d} | train_loss={train_loss:.4f}"
            if val_x is not None and val_y is not None:
                val_loss = self.compute_loss(val_x, val_y)
                val_pred = self.predict(val_x)
                val_f1 = macro_f1_score(val_y, val_pred, self.num_classes)
                history.val_loss.append(float(val_loss))
                history.val_macro_f1.append(float(val_f1))
                message += f" | val_loss={val_loss:.4f} | val_macro_f1={val_f1:.4f}"
            print(message)
        return history


class CustomTfidfVectorizerLab2:
    def __init__(self, tf_mode="raw", idf_mode="raw"):
        if tf_mode not in TF_MODES:
            raise ValueError(f"Unsupported tf_mode: {tf_mode}")
        if idf_mode not in IDF_MODES:
            raise ValueError(f"Unsupported idf_mode: {idf_mode}")
        self.tf_mode = tf_mode
        self.idf_mode = idf_mode
        self.vocabulary_ = {}
        self.idf_ = None
        self.most_frequent_corpus_word_ = None

    def fit(self, texts):
        document_frequency = Counter()
        corpus_frequency = Counter()

        for text in tqdm(texts, desc="Fitting vocabulary/IDF"):
            tokens = tokenize(text)
            if not tokens:
                continue
            term_counts = Counter(tokens)
            corpus_frequency.update(term_counts)
            document_frequency.update(term_counts.keys())

        self.vocabulary_ = {
            term: idx for idx, term in enumerate(sorted(document_frequency.keys()))
        }

        n_docs = len(texts)
        idf_raw = np.zeros(len(self.vocabulary_), dtype=np.float64)

        for term, index in self.vocabulary_.items():
            df = document_frequency[term]
            idf_raw[index] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        if not corpus_frequency:
            self.idf_ = idf_raw
            return self

        self.most_frequent_corpus_word_ = corpus_frequency.most_common(1)[0][0]

        if self.idf_mode == "raw":
            self.idf_ = idf_raw
        else:
            # Normalize IDF using the IDF range anchored by the most frequent corpus word.
            most_freq_idx = self.vocabulary_[self.most_frequent_corpus_word_]
            min_anchor = idf_raw[most_freq_idx]
            max_value = float(np.max(idf_raw))
            denominator = max(max_value - min_anchor, 1e-12)
            self.idf_ = (idf_raw - min_anchor) / denominator

        return self

    def _tf_weight(self, count, total_words, max_word_count):
        if self.tf_mode == "raw":
            return float(count)
        if self.tf_mode == "len_norm":
            return float(count) / max(float(total_words), 1.0)
        return float(count) / max(float(max_word_count), 1.0)

    def transform(self, texts):
        rows = []
        cols = []
        data = []

        for row_index, text in enumerate(tqdm(texts, desc="Vectorizing texts")):
            tokens = tokenize(text)
            if not tokens:
                continue
            counts = Counter(tokens)
            total_words = len(tokens)
            max_word_count = max(counts.values())

            entries = []
            for term, count in counts.items():
                if term not in self.vocabulary_:
                    continue
                col_index = self.vocabulary_[term]
                tf_value = self._tf_weight(count, total_words, max_word_count)
                value = tf_value * self.idf_[col_index]
                entries.append((col_index, value))

            if not entries:
                continue

            for col_index, value in entries:
                rows.append(row_index)
                cols.append(col_index)
                data.append(value)

        return csr_matrix(
            (np.array(data, dtype=np.float64), (np.array(rows), np.array(cols))),
            shape=(len(texts), len(self.vocabulary_)),
            dtype=np.float64,
        )

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


def confusion_matrix(true_labels, predicted_labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        matrix[true_label, predicted_label] += 1
    return matrix


def macro_f1_score(true_labels, predicted_labels, num_classes):
    matrix = confusion_matrix(true_labels, predicted_labels, num_classes)
    f1_scores = []
    for class_index in range(num_classes):
        tp = matrix[class_index, class_index]
        fp = matrix[:, class_index].sum() - tp
        fn = matrix[class_index, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / num_classes)


def accuracy_score(true_labels, predicted_labels):
    return float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels)))


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_comparison_csv(path, rows):
    headers = [
        "experiment",
        "tf_mode",
        "idf_mode",
        "validation_accuracy",
        "validation_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "vocabulary_size",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(name, tf_mode, idf_mode, train_rows, val_rows, test_rows, args):
    print("\n" + "=" * 80)
    print(f"Running {name}")
    print(f"TF : {TF_MODES[tf_mode]}")
    print(f"IDF: {IDF_MODES[idf_mode]}")
    print("=" * 80)

    train_texts = [row["text"] for row in train_rows]
    val_texts = [row["text"] for row in val_rows]
    test_texts = [row["text"] for row in test_rows]

    train_labels = [row["label"] for row in train_rows]
    val_labels = [row["label"] for row in val_rows]
    test_labels = [row["label"] for row in test_rows]

    label_encoder = LabelEncoder(LABELS)
    train_y = label_encoder.encode(train_labels)
    val_y = label_encoder.encode(val_labels)
    test_y = label_encoder.encode(test_labels)

    vectorizer = CustomTfidfVectorizerLab2(tf_mode=tf_mode, idf_mode=idf_mode)
    train_x = vectorizer.fit_transform(train_texts)
    val_x = vectorizer.transform(val_texts)
    test_x = vectorizer.transform(test_texts)

    model = LogisticRegressionOVR(
        num_features=train_x.shape[1],
        num_classes=len(LABELS),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
    )
    history = model.fit(train_x, train_y, val_x=val_x, val_y=val_y)

    val_pred = model.predict(val_x)
    test_pred = model.predict(test_x)

    result = {
        "experiment": name,
        "tf_mode": tf_mode,
        "idf_mode": idf_mode,
        "tf_description": TF_MODES[tf_mode],
        "idf_description": IDF_MODES[idf_mode],
        "most_frequent_corpus_word": vectorizer.most_frequent_corpus_word_,
        "vocabulary_size": int(train_x.shape[1]),
        "validation_accuracy": accuracy_score(val_y, val_pred),
        "validation_macro_f1": macro_f1_score(val_y, val_pred, len(LABELS)),
        "test_accuracy": accuracy_score(test_y, test_pred),
        "test_macro_f1": macro_f1_score(test_y, test_pred, len(LABELS)),
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_macro_f1": history.val_macro_f1,
        },
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Lab 2: TF-IDF normalization experiments")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lab1-output-dir",
        type=str,
        default="../LAB 1/outputs",
        help="Folder containing train.csv, validation.csv, test.csv from Lab 1",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    lab1_output_dir = (script_dir / args.lab1_output_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = lab1_output_dir / "train.csv"
    validation_csv = lab1_output_dir / "validation.csv"
    test_csv = lab1_output_dir / "test.csv"

    for required_file in (train_csv, validation_csv, test_csv):
        if not required_file.exists():
            raise FileNotFoundError(f"Required file not found: {required_file}")

    print("Loading existing Lab 1 dataset splits...")
    train_rows = read_rows(train_csv)
    val_rows = read_rows(validation_csv)
    test_rows = read_rows(test_csv)
    print(f"Loaded train={len(train_rows)}, validation={len(val_rows)}, test={len(test_rows)}")

    experiments = [
        ("exp_1_raw_tf_raw_idf", "raw", "raw"),
        ("exp_2_raw_tf_norm_idf", "raw", "maxword_norm"),
        ("exp_3_len_norm_tf_raw_idf", "len_norm", "raw"),
        ("exp_4_len_norm_tf_norm_idf", "len_norm", "maxword_norm"),
        ("exp_5_max_norm_tf_raw_idf", "max_norm", "raw"),
        ("exp_6_max_norm_tf_norm_idf", "max_norm", "maxword_norm"),
    ]

    all_results = []
    for name, tf_mode, idf_mode in experiments:
        result = run_experiment(name, tf_mode, idf_mode, train_rows, val_rows, test_rows, args)
        all_results.append(result)

    all_results.sort(key=lambda x: x["test_macro_f1"], reverse=True)

    comparison_rows = []
    for item in all_results:
        comparison_rows.append(
            {
                "experiment": item["experiment"],
                "tf_mode": item["tf_mode"],
                "idf_mode": item["idf_mode"],
                "validation_accuracy": f"{item['validation_accuracy']:.6f}",
                "validation_macro_f1": f"{item['validation_macro_f1']:.6f}",
                "test_accuracy": f"{item['test_accuracy']:.6f}",
                "test_macro_f1": f"{item['test_macro_f1']:.6f}",
                "vocabulary_size": item["vocabulary_size"],
            }
        )

    write_json(output_dir / "lab2_all_metrics.json", {"results": all_results})
    write_comparison_csv(output_dir / "lab2_experiment_comparison.csv", comparison_rows)

    best = all_results[0]
    summary = {
        "best_experiment": best["experiment"],
        "best_tf_mode": best["tf_mode"],
        "best_idf_mode": best["idf_mode"],
        "best_test_macro_f1": best["test_macro_f1"],
        "best_test_accuracy": best["test_accuracy"],
        "total_experiments": len(all_results),
    }
    write_json(output_dir / "lab2_summary.json", summary)

    print("\nCompleted all 6 experiments.")
    print("Top configurations by test macro-F1:")
    for rank, item in enumerate(all_results[:6], start=1):
        print(
            f"{rank}. {item['experiment']} | TF={item['tf_mode']} | IDF={item['idf_mode']} "
            f"| test_macro_f1={item['test_macro_f1']:.4f} | test_acc={item['test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
