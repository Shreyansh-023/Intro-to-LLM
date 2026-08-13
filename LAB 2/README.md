# Lab 2: TF-IDF Normalization Experiments

This lab reuses the dataset splits created in Lab 1 and runs 6 logistic regression experiments for all combinations of:

- TF variants (3):
  - `raw`: unnormalized TF (term count)
  - `len_norm`: TF normalized by total number of words in the sentence
  - `max_norm`: TF normalized by the most frequent word count in the sentence
- IDF variants (2):
  - `raw`: unnormalized IDF
  - `maxword_norm`: IDF normalized using the most frequent word across sentences

Total combinations: `3 x 2 = 6`.

## Input dataset

The script uses existing files from `../LAB 1/outputs/`:

- `train.csv`
- `validation.csv`
- `test.csv`

No dataset download is performed in Lab 2.

## Run

From the `LAB 2` folder:

```powershell
python lab2_tfidf_normalization_experiments.py
```

Optional hyperparameters:

```powershell
python lab2_tfidf_normalization_experiments.py --epochs 25 --learning-rate 0.5 --l2 0.0001
```

## Outputs

Generated in `LAB 2/outputs/`:

- `lab2_experiment_comparison.csv`: compact table with all 6 experiment scores
- `lab2_all_metrics.json`: full metrics + training history for each experiment
- `lab2_summary.json`: best configuration summary
