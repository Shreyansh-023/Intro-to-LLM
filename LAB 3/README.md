# Lab 3: IndicCorpV2 Dataset Preparation

This script prepares a monolingual LAB 3 dataset from `ai4bharat/IndicCorpV2` by:

1. Downloading one language split
2. Sentence-tokenizing paragraph rows
3. Building length-balanced dev/test splits (1000 each)
4. Creating train subsets for `100000`, `300000`, `500000`, and `1000000`

## File

- `lab3_prepare_dataset.py`
- `lab3_bpe_wordpiece_experiments.py`

## Run

From inside `LAB 3`:

```powershell
python lab3_prepare_dataset.py --language hin_Deva --output-dir outputs/hin_Deva
```

Optional args:

```powershell
python lab3_prepare_dataset.py --language hin_Deva --output-dir outputs/hin_Deva --seed 42 --min-tokens 3
```

## Run BPE + WordPiece experiments

After dataset preparation, run:

```powershell
python lab3_bpe_wordpiece_experiments.py --data-dir outputs/hin_Deva --output-dir outputs/hin_Deva/experiments
```

This runs the required settings:

- Train sizes: `100000`, `300000`, `500000`, `1000000`
- Vocabulary sizes: `20000`, `30000`, `50000`
- Algorithms: `BPE` and `WordPiece`

## Expected outputs

- `raw_sentences.txt`: collected sentence pool
- `raw_count.json`: count checkpoint
- `train_full.csv`: full training set (up to 1,000,000 rows)
- `train_100000.csv`
- `train_300000.csv`
- `train_500000.csv`
- `train_1000000.csv`
- `dev.csv` (1000 rows)
- `test.csv` (1000 rows)
- `split_summary.json`: length stats and bin distributions
- `experiments/lab3_tokenization_comparison.csv`: one row per experiment configuration
- `experiments/lab3_tokenization_report.json`: detailed metrics for dev/test tokenization
- `experiments/tokenizers/.../tokenizer.json`: trained tokenizer files for each setting
- `experiments/samples/*_samples.csv`: sample tokenized outputs to show tokenization differences

## Notes

- Use one language per run (`--language`).
- `dev` and `test` are sampled across sentence-length bins so they are not concentrated around similar lengths.
- Re-running with the same output directory reuses existing collected raw sentences if count is already sufficient.
- The experiment script requires the `tokenizers` package (`pip install tokenizers`).
