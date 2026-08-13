# Lab 1: Language Identification with IndicCorpV2

This lab builds a language identification system using the `ai4bharat/IndicCorpV2` corpus from Hugging Face.

## What the script does

1. Loads the 23 IndicCorpV2 language splits using the same labels as the repository:
   - `asm_Beng`, `ben_Beng`, `brx_Deva`, `doi_Deva`, `gom_Deva`, `guj_Gujr`, `hin_Deva`, `kan_Knda`, `kas_Arab`, `khasi`, `mai_Deva`, `mal_Mlym`, `mar_Deva`, `mni_Mtei`, `npi_Deva`, `ory_Orya`, `pan_Guru`, `san_Deva`, `santhali`, `snd_Deva`, `tam_Taml`, `tel_Telu`, `urd_Arab`
2. Extracts 1000 sentences per label by sentence-splitting the paragraph text.
3. Creates a stratified split:
   - Train: 800 per label
   - Validation: 100 per label
   - Test: 100 per label
4. Builds a custom TF-IDF vectorizer without using `sklearn`.
5. Trains a custom multiclass logistic regression classifier without using `sklearn`.
6. Evaluates with a custom macro-F1 implementation without using `sklearn`.

## Features used in TF-IDF

- Word unigrams
- Word bigrams
- Character 2-grams
- Character 3-grams
- Character 4-grams

## Files

- `lab1_language_identification.py`: end-to-end implementation
- `outputs/indiccorpv2_lid_full.csv`: full sampled dataset
- `outputs/train.csv`: train split
- `outputs/validation.csv`: validation split
- `outputs/test.csv`: test split
- `outputs/test_predictions.csv`: predictions on the test split
- `outputs/metrics.json`: training history and final metrics

## How to run

```powershell
python lab1_language_identification.py
```

Optional hyperparameters:

```powershell
python lab1_language_identification.py --epochs 30 --learning-rate 0.3 --l2 0.0001 --output-dir outputs
```

## Notes

- The dataset is streamed from Hugging Face, so the first run may take time depending on network speed.
- The sentence tokenizer is regex-based and handles common sentence-ending markers such as `.`, `!`, `?`, `।`, `॥`, and `؟`.
- No `sklearn` classes are used for TF-IDF, logistic regression, or macro-F1.
- Dataset collection is checkpointed per language inside `outputs/checkpoints/`. If the run stops because of a network handshake/read-timeout, rerunning the same command resumes from the saved checkpoint.
- The script also pauses after every 3 languages by default to reduce repeated Hugging Face handshake failures. You can change this with `--pause-every` and `--pause-seconds`.
