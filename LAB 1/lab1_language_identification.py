import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset
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


SENTENCE_BOUNDARY = re.compile(r"(?<=[\.\!\?।॥؟])\s+|\n+")
MULTISPACE = re.compile(r"\s+")


def normalize_text(text):
    text = MULTISPACE.sub(" ", text.strip())
    return text


def split_into_sentences(paragraph):
    normalized = normalize_text(paragraph)
    if not normalized:
        return []
    parts = SENTENCE_BOUNDARY.split(normalized)
    sentences = []
    for part in parts:
        cleaned = normalize_text(part)
        if len(cleaned) >= 5 and any(ch.isalpha() for ch in cleaned):
            sentences.append(cleaned)
    return sentences


def is_network_timeout_error(error):
    message = str(error).lower()
    return any(
        phrase in message
        for phrase in (
            "handshake operation timed out",
            "read timed out",
            "readtimeout",
            "ssl",
            "connectionpool",
            "max retries exceeded",
            "temporarily unavailable",
        )
    )


def load_checkpoint(checkpoint_path, target_count):
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("target_count") != target_count:
        return None
    return state


def save_checkpoint(checkpoint_path, label, target_count, seen, collected, buffered):
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "label": label,
                "target_count": target_count,
                "seen_paragraphs": seen,
                "collected": collected,
                "buffer": buffered,
                "saved_at_epoch": time.time(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def sample_sentences_from_label(
    label,
    target_count,
    seed,
    checkpoint_dir,
    log_every=250,
    max_retries=5,
    retry_wait_seconds=20,
):
    checkpoint_path = checkpoint_dir / f"{label}.json"
    checkpoint_state = load_checkpoint(checkpoint_path, target_count)
    if checkpoint_state is not None and len(checkpoint_state.get("collected", [])) >= target_count:
        collected = checkpoint_state["collected"][:target_count]
        print(f"\n[{label}] Reusing completed checkpoint with {len(collected)} sentences.")
        return collected

    collected = []
    seen = 0
    buffer = []
    rng = random.Random(seed)
    next_milestone = min(log_every, target_count)

    if checkpoint_state is not None:
        collected = checkpoint_state.get("collected", [])
        seen = checkpoint_state.get("seen_paragraphs", 0)
        buffer = checkpoint_state.get("buffer", [])
        while next_milestone <= len(collected):
            next_milestone += log_every
        print(
            f"\n[{label}] Resuming from checkpoint: {len(collected)}/{target_count} sentences, "
            f"{seen} paragraphs already scanned."
        )
    else:
        print(f"\n[{label}] Starting stream collection for {target_count} sentences...")

    attempt = 1
    while attempt <= max_retries:
        try:
            print(f"[{label}] Opening streaming dataset connection (attempt {attempt})...")
            dataset = load_dataset("ai4bharat/IndicCorpV2", split=label, streaming=True)

            fast_forwarded = 0
            if seen > 0:
                print(f"[{label}] Fast-forwarding {seen} previously scanned paragraphs...")

            for example in dataset:
                if fast_forwarded < seen:
                    fast_forwarded += 1
                    if fast_forwarded % log_every == 0 or fast_forwarded == seen:
                        print(f"[{label}] fast-forwarded {fast_forwarded}/{seen} paragraphs")
                    continue

                paragraph = example.get("text", "")
                if not paragraph:
                    continue

                seen += 1
                sentences = split_into_sentences(paragraph)
                if not sentences:
                    if seen % log_every == 0:
                        print(
                            f"[{label}] scanned {seen} paragraphs | collected {len(collected)} sentences | "
                            f"buffered {len(buffer)}"
                        )
                        save_checkpoint(checkpoint_path, label, target_count, seen, collected, buffer)
                    continue

                rng.shuffle(sentences)
                for sentence in sentences:
                    buffer.append(sentence)
                    if len(buffer) >= 256:
                        rng.shuffle(buffer)
                        while buffer and len(collected) < target_count:
                            collected.append(buffer.pop())
                        save_checkpoint(checkpoint_path, label, target_count, seen, collected, buffer)

                    if len(collected) >= next_milestone:
                        print(
                            f"[{label}] reached {len(collected)}/{target_count} sentences "
                            f"after scanning {seen} paragraphs"
                        )
                        while next_milestone <= len(collected):
                            next_milestone += log_every
                        save_checkpoint(checkpoint_path, label, target_count, seen, collected, buffer)

                    if len(collected) >= target_count:
                        save_checkpoint(checkpoint_path, label, target_count, seen, collected, buffer)
                        print(
                            f"[{label}] Completed with {len(collected)} sentences from {seen} paragraphs."
                        )
                        return collected[:target_count]

                if seen % log_every == 0:
                    print(
                        f"[{label}] scanned {seen} paragraphs | collected {len(collected)} sentences | "
                        f"buffered {len(buffer)}"
                    )
                    save_checkpoint(checkpoint_path, label, target_count, seen, collected, buffer)
            break
        except Exception as error:
            save_checkpoint(checkpoint_path, label, target_count, seen, collected, buffer)
            if not is_network_timeout_error(error) or attempt >= max_retries:
                raise
            wait_seconds = retry_wait_seconds * attempt
            print(
                f"[{label}] Network timeout/handshake issue. Checkpoint saved at "
                f"{len(collected)} sentences and {seen} scanned paragraphs."
            )
            print(f"[{label}] Waiting {wait_seconds} seconds before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_seconds)
            attempt += 1

    rng.shuffle(buffer)
    for sentence in buffer:
        if len(collected) >= target_count:
            break
        collected.append(sentence)

    save_checkpoint(checkpoint_path, label, target_count, seen, collected, buffer)
    if len(collected) < target_count:
        raise RuntimeError(
            f"Could only collect {len(collected)} sentences for {label}, expected {target_count}."
        )
    print(f"[{label}] Completed with {len(collected)} sentences from {seen} paragraphs.")
    return collected[:target_count]


def build_dataset(labels, samples_per_label, seed, output_dir, pause_every=3, pause_seconds=15):
    rows = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Preparing dataset for {len(labels)} labels with {samples_per_label} sentences per label "
        f"({len(labels) * samples_per_label} total samples)..."
    )
    for index, label in enumerate(labels, start=1):
        print(f"\n=== [{index}/{len(labels)}] Processing label: {label} ===")
        sentences = sample_sentences_from_label(label, samples_per_label, seed, checkpoint_dir)
        for sentence in sentences:
            rows.append({"text": sentence, "label": label})
        print(f"[{label}] Added {len(sentences)} samples. Running total: {len(rows)}")
        if index % pause_every == 0 and index != len(labels):
            print(
                f"\nCompleted {index} languages. Cooling down for {pause_seconds} seconds "
                f"before opening the next Hugging Face connection..."
            )
            time.sleep(pause_seconds)
    print("\nDataset creation finished.")
    return rows


def stratified_split(rows, labels, seed):
    rng = random.Random(seed)
    train_rows = []
    val_rows = []
    test_rows = []

    for label in labels:
        label_rows = [row for row in rows if row["label"] == label]
        rng.shuffle(label_rows)
        train_rows.extend(label_rows[:800])
        val_rows.extend(label_rows[800:900])
        test_rows.extend(label_rows[900:1000])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def save_rows_to_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


class CustomTfidfVectorizer:
    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = None

    @staticmethod
    def word_ngrams(text):
        tokens = text.split()
        features = []
        for token in tokens:
            features.append(("w1", token))
        for i in range(len(tokens) - 1):
            features.append(("w2", tokens[i] + " " + tokens[i + 1]))
        return features

    @staticmethod
    def char_ngrams(text):
        features = []
        for n in (2, 3, 4):
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                features.append((f"c{n}", text[i : i + n]))
        return features

    def extract_features(self, text):
        normalized = normalize_text(text)
        return self.word_ngrams(normalized) + self.char_ngrams(normalized)

    def fit(self, texts):
        document_frequency = Counter()
        for text in tqdm(texts, desc="Fitting TF-IDF vocabulary"):
            unique_features = set(self.extract_features(text))
            document_frequency.update(unique_features)

        self.vocabulary_ = {
            feature: index for index, feature in enumerate(sorted(document_frequency.keys()))
        }
        n_docs = len(texts)
        self.idf_ = np.zeros(len(self.vocabulary_), dtype=np.float64)

        for feature, index in self.vocabulary_.items():
            df = document_frequency[feature]
            self.idf_[index] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self

    def transform(self, texts):
        rows = []
        cols = []
        data = []

        for row_index, text in enumerate(tqdm(texts, desc="Transforming texts")):
            counts = Counter(self.extract_features(text))
            if not counts:
                continue
            total = float(sum(counts.values()))
            norm = 0.0
            entries = []
            for feature, count in counts.items():
                if feature not in self.vocabulary_:
                    continue
                col_index = self.vocabulary_[feature]
                tf = count / total
                value = tf * self.idf_[col_index]
                entries.append((col_index, value))
                norm += value * value

            if not entries:
                continue

            norm = math.sqrt(norm)
            if norm == 0.0:
                norm = 1.0
            for col_index, value in entries:
                rows.append(row_index)
                cols.append(col_index)
                data.append(value / norm)

        return csr_matrix(
            (np.array(data, dtype=np.float64), (np.array(rows), np.array(cols))),
            shape=(len(texts), len(self.vocabulary_)),
            dtype=np.float64,
        )

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


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
    def __init__(self, num_features, num_classes, learning_rate=0.5, epochs=25, l2=1e-4):
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
        negative_log_likelihood = -np.log(clipped[np.arange(len(labels)), labels]).mean()
        regularization = 0.5 * self.l2 * np.sum(self.weights * self.weights)
        return negative_log_likelihood + regularization

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


def write_metrics(output_path, metrics):
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def write_predictions(output_path, rows, predicted_labels):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "gold_label", "predicted_label"])
        writer.writeheader()
        for row, predicted_label in zip(rows, predicted_labels):
            writer.writerow(
                {
                    "text": row["text"],
                    "gold_label": row["label"],
                    "predicted_label": predicted_label,
                }
            )


def run_pipeline(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset from IndicCorpV2...")
    rows = build_dataset(
        LABELS,
        args.samples_per_label,
        args.seed,
        output_dir,
        pause_every=args.pause_every,
        pause_seconds=args.pause_seconds,
    )
    train_rows, val_rows, test_rows = stratified_split(rows, LABELS, args.seed)
    print(
        f"Saved stratified splits with train={len(train_rows)}, "
        f"validation={len(val_rows)}, test={len(test_rows)}"
    )

    save_rows_to_csv(rows, output_dir / "indiccorpv2_lid_full.csv")
    save_rows_to_csv(train_rows, output_dir / "train.csv")
    save_rows_to_csv(val_rows, output_dir / "validation.csv")
    save_rows_to_csv(test_rows, output_dir / "test.csv")

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

    vectorizer = CustomTfidfVectorizer()
    print("\nVectorizing train split...")
    train_x = vectorizer.fit_transform(train_texts)
    print("Vectorizing validation split...")
    val_x = vectorizer.transform(val_texts)
    print("Vectorizing test split...")
    test_x = vectorizer.transform(test_texts)

    model = LogisticRegressionOVR(
        num_features=train_x.shape[1],
        num_classes=len(LABELS),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
    )
    print("\nTraining custom logistic regression model...")
    history = model.fit(train_x, train_y, val_x=val_x, val_y=val_y)

    print("\nRunning final evaluation...")
    val_pred = model.predict(val_x)
    test_pred = model.predict(test_x)

    metrics = {
        "labels": LABELS,
        "train_size": len(train_rows),
        "validation_size": len(val_rows),
        "test_size": len(test_rows),
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

    write_metrics(output_dir / "metrics.json", metrics)
    write_predictions(output_dir / "test_predictions.csv", test_rows, label_encoder.decode(test_pred))

    print("\nFinal Results")
    print(f"Train size      : {metrics['train_size']}")
    print(f"Validation size : {metrics['validation_size']}")
    print(f"Test size       : {metrics['test_size']}")
    print(f"Vocabulary size : {metrics['vocabulary_size']}")
    print(f"Validation Macro-F1: {metrics['validation_macro_f1']:.4f}")
    print(f"Test Macro-F1      : {metrics['test_macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Lab 1: IndicCorpV2 language identification")
    parser.add_argument("--samples-per-label", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--pause-every", type=int, default=3)
    parser.add_argument("--pause-seconds", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_pipeline(arguments)
