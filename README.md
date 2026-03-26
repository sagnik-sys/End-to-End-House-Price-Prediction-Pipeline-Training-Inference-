# 🏠 End-to-End House Price Prediction Pipeline

A clean, self-contained machine learning pipeline for predicting California house prices — covering the full lifecycle from **data preprocessing** and **model training** to **batch inference**, all in a single Python script.

---

## Overview

This project uses the [California Housing dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) to build a `RandomForestRegressor` model wrapped inside a reusable `sklearn` pipeline. On first run, it trains and serializes the model. On every subsequent run, it loads the saved model and runs inference on new input data, writing predictions to `output.csv`.

The key design idea: **the same script handles both training and inference**, determined automatically by whether a saved model file already exists on disk.

---

## How It Works

```
First Run (no model.pkl present)
  └─ Load housing.csv
  └─ Stratified train/test split (on income category)
  └─ Save test set → input.csv
  └─ Build preprocessing pipeline (imputation + scaling + one-hot encoding)
  └─ Train RandomForestRegressor
  └─ Serialize model → model.pkl and pipeline → pipeline.pkl
  └─ Print: "Model is trained."

Subsequent Runs (model.pkl exists)
  └─ Load model.pkl and pipeline.pkl
  └─ Read input.csv
  └─ Transform features via saved pipeline
  └─ Predict median_house_value
  └─ Save results → output.csv
  └─ Print: "Inference is complete, results saved to output.csv"
```

---

## Project Structure

```
.
├── housing.csv          # Raw California housing dataset (training source)
├── input.csv            # Auto-generated test set (used for inference)
├── input - Copy.csv     # Backup / alternate inference input
├── output.csv           # Inference results with predicted house values
├── main.py              # Main pipeline script (training + inference)
├── main_old.py          # Earlier iteration of the pipeline
├── model.pkl            # Serialized trained model (auto-generated)
├── pipeline.pkl         # Serialized preprocessing pipeline (auto-generated)
└── .gitattributes
```

---

## Prerequisites

- Python 3.8+
- pip

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

Or create a `requirements.txt` file in the project root with the following contents:

```
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
joblib>=1.3.0
```

Then install all at once:

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | ≥ 1.3.0 | Pipeline, preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`), `RandomForestRegressor`, `StratifiedShuffleSplit`, RMSE, cross-validation |
| `numpy` | ≥ 1.24.0 | Numerical operations, `np.inf` for income binning with `pd.cut` |
| `pandas` | ≥ 2.0.0 | Loading CSVs (`housing.csv`, `input.csv`), feature manipulation, writing `output.csv` |
| `joblib` | ≥ 1.3.0 | Serializing and loading `model.pkl` and `pipeline.pkl` |

---

## Usage

### Step 1 — Train the model

Run the script with `housing.csv` present and no `model.pkl` in the directory:

```bash
python main.py
```

This will:
- Split the data into training and test sets using stratified sampling on `median_income`
- Fit the preprocessing pipeline and `RandomForestRegressor`
- Save `model.pkl`, `pipeline.pkl`, and `input.csv` to disk

### Step 2 — Run inference

On any subsequent run (with `model.pkl` present):

```bash
python main.py
```

The script loads the saved model and pipeline, transforms `input.csv`, and writes predictions to `output.csv`.

### Step 3 — Provide custom input

To predict prices for new houses, replace (or update) `input.csv` with your own data. Make sure the column schema matches the training data:

| Column | Description |
|---|---|
| `longitude` | Geographic coordinate |
| `latitude` | Geographic coordinate |
| `housing_median_age` | Median age of houses in block |
| `total_rooms` | Total rooms in block |
| `total_bedrooms` | Total bedrooms in block |
| `population` | Population of block |
| `households` | Number of households |
| `median_income` | Median household income (in tens of thousands) |
| `ocean_proximity` | Categorical: `NEAR BAY`, `<1H OCEAN`, `INLAND`, etc. |

Then run `python main.py` — predictions will appear in `output.csv` as a new `median_house_value` column.

---

## Pipeline Details

### Preprocessing

The pipeline handles both numerical and categorical features separately using `ColumnTransformer`:

- **Numerical features**: Missing values filled with the column median (`SimpleImputer`), then standardized (`StandardScaler`)
- **Categorical feature** (`ocean_proximity`): One-hot encoded (`OneHotEncoder` with `handle_unknown="ignore"`)

### Model

`RandomForestRegressor` with `random_state=42` for reproducibility. The model is evaluated using RMSE and cross-validation during development.

### Persistence

Both the fitted model and the fitted preprocessing pipeline are serialized with `joblib`, ensuring the exact same transformations are applied at inference time as during training — preventing data leakage and training/serving skew.

---

## Resetting the Pipeline

To retrain from scratch, delete the saved artifacts and re-run:

```bash
rm model.pkl pipeline.pkl
python main.py
```

---

## Author

**[sagnik-sys](https://github.com/sagnik-sys)**

---

## License

This project is open source. Feel free to fork, extend, and build on it.
