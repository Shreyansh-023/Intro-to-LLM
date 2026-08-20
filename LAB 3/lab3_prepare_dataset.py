import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset


SENTENCE_BOUNDARY = re.compile(r"(?<=[\.\!\?।॥؟])\s+|\n+")
MULTISPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return MULTISPACE.sub(" ", text.strip())


def split_into_sentences(paragraph: str) -> list[str]:
    normalized = normalize_text(paragraph)
    if not normalized:
        return []

    parts = SENTENCE_BOUNDARY.split(normalized)
    sentences = []
    for part in parts:
        cleaned = normalize_text(part)
        if cleaned and any(ch.isalpha() for ch in cleaned):
            sentences.append(cleaned)
    return sentences


def token_length(sentence: str) -> int:
    return len(sentence.split())


def allocate_quotas(bin_counts: np.ndarray, target_size: int) -> np.ndarray:
    total = int(bin_counts.sum())
    if total < target_size:
        raise ValueError(f"Not enough data to allocate {target_size} samples.")

    raw = (bin_counts / total) * target_size
    quotas = np.floor(raw).astype(int)
    quotas = np.minimum(quotas, bin_counts)

    remainder = target_size - int(quotas.sum())
    fractional = raw - np.floor(raw)

    while remainder > 0:
        capacity = bin_counts - quotas
        candidates = np.where(capacity > 0)[0]
        if len(candidates) == 0:
            break
        best = candidates[np.argmax(fractional[candidates])]
        quotas[best] += 1
        remainder -= 1

    if int(quotas.sum()) != target_size:
        raise ValueError("Failed to allocate exact target size across bins.")

    return quotas


def select_indices_by_bins(
    lengths: np.ndarray,
    target_size: int,
    rng: np.random.Generator,
    boundaries: np.ndarray,
    blocked: set[int] | None = None,
) -> set[int]:
    if blocked is None:
        blocked = set()

    bin_ids = np.digitize(lengths, boundaries, right=True)
    n_bins = int(bin_ids.max()) + 1

    available_mask = np.ones(len(lengths), dtype=bool)
    if blocked:
        blocked_idx = np.array(sorted(blocked), dtype=int)
        available_mask[blocked_idx] = False

    bin_counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        bin_counts[b] = int(np.sum((bin_ids == b) & available_mask))

    quotas = allocate_quotas(bin_counts, target_size)

    selected = set()
    for b in range(n_bins):
        needed = int(quotas[b])
        if needed == 0:
            continue

        candidates = np.where((bin_ids == b) & available_mask)[0]
        chosen = rng.choice(candidates, size=needed, replace=False)
        for idx in chosen.tolist():
            selected.add(int(idx))
            available_mask[idx] = False

    if len(selected) != target_size:
        raise ValueError("Could not sample the requested number of indices.")

    return selected


def collect_sentences(
    language: str,
    output_dir: Path,
    required_count: int,
    seed: int,
    min_tokens: int,
) -> Path:
    raw_path = output_dir / "raw_sentences.txt"
    count_path = output_dir / "raw_count.json"

    existing = 0
    if raw_path.exists() and count_path.exists():
        with count_path.open("r", encoding="utf-8") as f:
            existing = int(json.load(f).get("count", 0))

    if existing >= required_count:
        return raw_path

    mode = "a" if raw_path.exists() else "w"
    rng = random.Random(seed)
    buffer: list[str] = []

    with raw_path.open(mode, encoding="utf-8") as out:
        dataset = load_dataset("ai4bharat/IndicCorpV2", split=language, streaming=True)

        collected = existing
        for row in dataset:
            text = row.get("text", "")
            if not text:
                continue

            sentences = split_into_sentences(text)
            if not sentences:
                continue

            rng.shuffle(sentences)
            for sentence in sentences:
                if token_length(sentence) < min_tokens:
                    continue

                buffer.append(sentence)
                if len(buffer) >= 2048:
                    rng.shuffle(buffer)
                    while buffer and collected < required_count:
                        out.write(buffer.pop() + "\n")
                        collected += 1

                if collected >= required_count:
                    break

            if collected >= required_count:
                break

        rng.shuffle(buffer)
        while buffer and collected < required_count:
            out.write(buffer.pop() + "\n")
            collected += 1

    with count_path.open("w", encoding="utf-8") as f:
        json.dump({"count": collected}, f, indent=2)

    if collected < required_count:
        raise RuntimeError(
            f"Collected only {collected} sentences; need {required_count}."
        )

    return raw_path


def read_lengths(raw_path: Path) -> np.ndarray:
    lengths = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            sentence = line.rstrip("\n")
            if not sentence:
                continue
            lengths.append(token_length(sentence))
    return np.array(lengths, dtype=np.int32)


def write_splits(
    raw_path: Path,
    output_dir: Path,
    dev_indices: set[int],
    test_indices: set[int],
    max_train_size: int,
    train_sizes: list[int],
) -> None:
    train_full_path = output_dir / "train_full.csv"
    dev_path = output_dir / "dev.csv"
    test_path = output_dir / "test.csv"

    with train_full_path.open("w", newline="", encoding="utf-8") as train_f, \
        dev_path.open("w", newline="", encoding="utf-8") as dev_f, \
        test_path.open("w", newline="", encoding="utf-8") as test_f:

        fieldnames = ["text", "length_tokens"]
        train_writer = csv.DictWriter(train_f, fieldnames=fieldnames)
        dev_writer = csv.DictWriter(dev_f, fieldnames=fieldnames)
        test_writer = csv.DictWriter(test_f, fieldnames=fieldnames)

        train_writer.writeheader()
        dev_writer.writeheader()
        test_writer.writeheader()

        with raw_path.open("r", encoding="utf-8") as raw_f:
            for idx, line in enumerate(raw_f):
                sentence = line.rstrip("\n")
                if not sentence:
                    continue

                row = {"text": sentence, "length_tokens": token_length(sentence)}
                if idx in dev_indices:
                    dev_writer.writerow(row)
                elif idx in test_indices:
                    test_writer.writerow(row)
                else:
                    train_writer.writerow(row)

    # Create nested train subsets from the start of shuffled train_full.
    for size in train_sizes:
        if size > max_train_size:
            continue

        subset_path = output_dir / f"train_{size}.csv"
        with train_full_path.open("r", encoding="utf-8") as src, subset_path.open(
            "w", newline="", encoding="utf-8"
        ) as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=["text", "length_tokens"])
            writer.writeheader()

            written = 0
            for row in reader:
                writer.writerow(row)
                written += 1
                if written >= size:
                    break


def summarize_lengths(
    lengths: np.ndarray,
    dev_indices: set[int],
    test_indices: set[int],
    output_dir: Path,
    boundaries: np.ndarray,
) -> None:
    all_idx = np.arange(len(lengths))
    dev_idx = np.array(sorted(dev_indices), dtype=int)
    test_idx = np.array(sorted(test_indices), dtype=int)
    train_mask = np.ones(len(lengths), dtype=bool)
    train_mask[dev_idx] = False
    train_mask[test_idx] = False

    def describe(arr: np.ndarray) -> dict:
        return {
            "count": int(len(arr)),
            "min": int(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "max": int(np.max(arr)),
            "mean": float(np.mean(arr)),
        }

    bin_ids = np.digitize(lengths, boundaries, right=True)
    train_bins = np.bincount(bin_ids[train_mask])
    dev_bins = np.bincount(bin_ids[dev_idx])
    test_bins = np.bincount(bin_ids[test_idx])

    summary = {
        "length_boundaries": boundaries.tolist(),
        "train": describe(lengths[train_mask]),
        "dev": describe(lengths[dev_idx]),
        "test": describe(lengths[test_idx]),
        "bin_counts": {
            "train": train_bins.tolist(),
            "dev": dev_bins.tolist(),
            "test": test_bins.tolist(),
        },
        "total_sentences": int(len(all_idx)),
    }

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare IndicCorpV2 LAB 3 splits with length-balanced dev/test sets."
    )
    parser.add_argument("--language", required=True, help="IndicCorpV2 split name, e.g., hin_Deva")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--max-train-size", type=int, default=1_000_000)
    parser.add_argument("--min-tokens", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_required = args.max_train_size + args.dev_size + args.test_size
    raw_path = collect_sentences(
        language=args.language,
        output_dir=output_dir,
        required_count=total_required,
        seed=args.seed,
        min_tokens=args.min_tokens,
    )

    lengths = read_lengths(raw_path)
    if len(lengths) < total_required:
        raise RuntimeError(f"Found only {len(lengths)} usable sentences in raw file.")

    # Use decile boundaries so dev/test are sampled across short/medium/long sentences.
    quantiles = np.linspace(0.1, 0.9, 9)
    boundaries = np.unique(np.quantile(lengths, quantiles))

    rng = np.random.default_rng(args.seed)
    dev_indices = select_indices_by_bins(lengths, args.dev_size, rng, boundaries)
    test_indices = select_indices_by_bins(
        lengths,
        args.test_size,
        rng,
        boundaries,
        blocked=dev_indices,
    )

    train_sizes = [100_000, 300_000, 500_000, 1_000_000]
    write_splits(
        raw_path=raw_path,
        output_dir=output_dir,
        dev_indices=dev_indices,
        test_indices=test_indices,
        max_train_size=args.max_train_size,
        train_sizes=train_sizes,
    )

    summarize_lengths(lengths, dev_indices, test_indices, output_dir, boundaries)

    print("Dataset preparation complete.")
    print(f"Language: {args.language}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
