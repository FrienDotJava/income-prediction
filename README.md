# Income Prediction (Adult Census)

> A modular machine-learning project that predicts whether a person’s annual income is **> $50K** or **≤ $50K**, built with a clean Cookiecutter Data Science structure, tracked with **DVC** (and DVCLive), and instrumented for **MLflow** experiment logging. A small **Streamlit** dashboard is included for quick exploration.

---

Try it here: https://income-prediction1.streamlit.app/

## 📌 Project goals

- Train solid baseline & boosted tree models for the Adult/Census Income task (binary classification >$50K).
- Keep work **reproducible** (DVC pipelines + parameters), **trackable** (DVCLive/MLflow), and **organized** (Cookiecutter DS layout).
- Provide a minimal **dashboard** to poke the model and visualize results.

---

## 🗂 Repository structure

```
├── data/                 # raw/ → interim/ → processed/ (DVC-managed)
├── docs/                 # (optional) project docs
├── dvclive/              # live metrics/artifacts from runs
├── income_classification/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py        # data download/prepare helpers
│   ├── features.py       # feature engineering
│   └── modeling/
│       ├── __init__.py
│       ├── predict.py    # inference script
│       └── train.py      # training script
├── notebooks/            # EDA & scratch work
├── references/           # data dictionary, notes, etc.
├── dashboard.py          # Streamlit mini app
├── dvc.yaml              # DVC pipeline (stages & deps)
├── dvc.lock              # DVC lockfile (auto-generated)
├── params.yaml           # central hyperparams & config
├── requirements.txt      # Python dependencies
├── Makefile              # convenience commands
└── README.md
```

---

## 📦 Dataset

This project uses the **Adult (Census Income)** dataset: **48,842** rows, **14** features, binary target (> $50K).  
You can obtain it from UCI or Kaggle.

> Place raw files under `data/raw/`.

---

## 🛠️ Quickstart

### 1) Setup environment

```bash
git clone https://github.com/FrienDotJava/income-prediction.git
cd income-prediction

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Get the data

Download the Adult/Census Income data and put it here:

```
data/
└── raw/
    └── adult.csv
```

### 3) Reproduce the pipeline (DVC)

```bash
dvc repro
```

- **Stages** and dependencies live in `dvc.yaml`; `dvc repro` runs data prep → feature building → training → evaluation.
- Metrics and plots are logged in `dvclive/`.

### 4) Tweak parameters & rerun

Edit **`params.yaml`** to change model settings, then:

```bash
dvc repro
```

### 5) Track experiments (MLflow)

```bash
mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore experiments.

### 6) Run the dashboard

```bash
streamlit run dashboard.py
```

---

## 🧪 Pipeline overview

- **Data prep**: clean & split the Adult dataset.  
- **Feature engineering**: encode categoricals, scale numerics.  
- **Model training**: Logistic Regression, RandomForest, GradientBoosting, etc.  
- **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.  
- **Experiment tracking**: DVC, DVCLive, MLflow.

---

## 📈 Example results

Typical accuracy: **80–86%** (depends on preprocessing and model).

---

## ▶️ Makefile shortcuts

```bash
make train      # Run training
make clean      # Clean temp artifacts
make dashboard  # Launch Streamlit dashboard
```

---

## 📚 References

- Adult (Census Income) dataset (UCI)
- Kaggle: Adult Census Income
- Cookiecutter Data Science

---

## 💡 Tips

- If only `params.yaml` changes, rerun `dvc repro`.  
- Use `dvc commit && dvc push` to sync data to remote storage.  
- Use Streamlit for fast visual validation.
