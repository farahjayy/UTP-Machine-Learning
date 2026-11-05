<div align="center">

# UTP Machine Learning Lab

**Eight practical milestones, from feature engineering to deep learning.**<br>
Worked Jupyter notebooks, the datasets they read, and the models they save.

<p>
  <img src="https://img.shields.io/badge/Labs_done-6_of_8-105666?style=for-the-badge" alt="6 of 8 labs done"> <img src="https://img.shields.io/badge/Datasets-12-d3968c?style=for-the-badge" alt="12 datasets"> <img src="https://img.shields.io/badge/Saved_models-5-2ea44f?style=for-the-badge" alt="5 saved models">
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3776ab?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-f37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/scikit--learn-f7931e?style=flat&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/PyTorch-ee4c2c?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=flat&logoColor=white" alt="Matplotlib">
</p>

<sub>Machine Learning Lab · Universiti Teknologi PETRONAS · Final Year, Semester 3</sub>

</div>

---

This repository holds my worked solutions for the Machine Learning practical milestones. Every notebook follows the same habits: split the data before anything else, fit preprocessing inside a pipeline so nothing leaks from the test set, tune with cross-validation, touch the test set once, and save what is worth reusing.

Repository: [github.com/farahjayy/UTP-Machine-Learning](https://github.com/farahjayy/UTP-Machine-Learning)

## Progress

| Lab | Topic | Notebook | Status |
|---|---|---|---|
| 1 | Feature engineering and feature selection | [`lab01_features_worked.ipynb`](all_experiments/lab01_features_worked.ipynb) | Done |
| 2 | Supervised learning I: regression, classification, and the workflow | [`lab02_supervised_workflow_worked.ipynb`](all_experiments/lab02_supervised_workflow_worked.ipynb) | Done |
| 3 | Supervised learning II: model families | [`lab03_supervised_models_worked.ipynb`](all_experiments/lab03_supervised_models_worked.ipynb) | Done |
| 4 | Unsupervised learning I: clustering and PCA | [`lab04_clustering_pca_worked.ipynb`](all_experiments/lab04_clustering_pca_worked.ipynb) | Done |
| 5 | Unsupervised learning II: anomaly detection and limited labels | [`lab05_anomaly_limited_labels_worked.ipynb`](all_experiments/lab05_anomaly_limited_labels_worked.ipynb) | Done |
| 6 | Reinforcement learning | [`lab06_reinforcement_worked.ipynb`](all_experiments/lab06_reinforcement_worked.ipynb) | Done |
| 7 | Deep learning: CNN | `lab07_deep_learning_cnn_worked.ipynb` | Not started |
| 8 | Deep learning: LSTM | `lab08_deep_learning_lstm_worked.ipynb` | Not started |

What each finished lab covers:

- **Lab 1:** Titanic feature engineering, scaling and distances on breast-cancer data, feature selection with `SelectKBest` and lasso, and a deliberate leakage demonstration on student grades.
- **Lab 2:** OLS vs ridge, date-ordered validation for the bike-sharing time series, baselines and decision thresholds, tuning with CV then testing once, and saving models with a model card.
- **Lab 3:** KNN vs Gaussian naive Bayes, decision-tree depth, random forest vs gradient boosting, and SVM kernels.
- **Lab 4:** Choosing K with elbow and silhouette, K-means vs DBSCAN, and PCA judged by a downstream classifier.
- **Lab 5:** Isolation Forest fraud ranking, then self-training and active learning with few labels.
- **Lab 6:** Tabular Q-learning in a 4×4 grid world, and reading the policy from the saved Q-table.

## Project layout

```
UTP-MACHINE-LEARNING/
├── all_datasets/        the 12 datasets (CSV, plus the digits ZIP, kept intact)
├── all_experiments/     one worked notebook per lab, plus predict_passenger.py
├── models/              models, model cards and the Q-table saved by the notebooks
├── results/lab01 … lab08/   evidence: every figure a lab draws, saved as figNN.png
└── requirements.txt     Python packages
```

Notebooks live in `all_experiments/` and find everything through `PROJECT_ROOT = Path.cwd().parent`, so **always start Jupyter from inside `all_experiments/`**. For a `FileNotFoundError`, check the working directory and the file name rather than moving files into the project root.

Each notebook's setup cell defines a small `save_fig()` helper that writes the current figure to `results/labNN/` just before it is shown. Running a notebook from top to bottom regenerates its figures in order (`fig01.png`, `fig02.png`, …). `results/` currently holds Labs 1–6; Labs 7 and 8 will add theirs.

## Datasets

Download each dataset from Kaggle and extract the files into `all_datasets/`. Keep the file names exactly as below.

| File | Where it is used |
|---|---|
| `breast_cancer_dataset.csv` | Lab 1 scaling; Lab 2 thresholds; Lab 3 KNN, naive Bayes, SVM; Lab 5 limited labels |
| `medical_cost_personal_dataset.csv` | Lab 1 pipelines; Lab 2 linear and ridge regression |
| `student_performance_dataset.csv` | Lab 1 data quality, prediction timing, feature selection |
| `titanic_dataset.csv` | Lab 1 feature engineering; Lab 2 baselines, tuning, saving; Lab 3 trees |
| `bike_sharing_dataset_day.csv`, `bike_sharing_dataset_hour.csv` | Lab 2 daily and hourly time splits; Lab 8 LSTM forecasting |
| `heart_failure_dataset.csv` | Lab 2 "Try it" exercise: mixed-type classification |
| `ames_housing_dataset.csv` | Lab 3 random forests and gradient boosting |
| `red_wine_quality_dataset.csv` | Lab 3 ensembles and feature importance |
| `mall_customers_dataset.csv` | Lab 4 K-means and DBSCAN |
| `optical_recognition_handwritten_digits.zip` | Lab 4 PCA; Lab 7 CNNs |
| `credit_card_fraud_dataset.csv` | Lab 5 anomaly detection and imbalanced evaluation |

## Set up

Use one environment per project, activate it before installing, and install once. Both routes use the same `requirements.txt` (`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`, `jupyter`, `torch`). PyTorch is only needed for Labs 7–8 and a CPU is enough.

### Route A: Python and venv

```bash
git clone git@github.com:farahjayy/UTP-Machine-Learning.git
cd UTP-Machine-Learning

# Windows
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Next session, activate `.venv` again; do not reinstall.

### Route B: Anaconda or Miniconda

```bash
conda create -n ml python=3.12
conda activate ml
pip install -r requirements.txt
```

Install with `pip`, not a mix of conda and pip, and never into `(base)`. If Jupyter cannot import a package, activate `ml`, restart Jupyter, and run Lab 1's environment cell to check the kernel.

### Run the notebooks

```bash
cd all_experiments
jupyter lab
```

Open a `labNN_*_worked.ipynb` and choose **Run → Run All Cells**. Notebooks that save a model create `models/` themselves if it is missing. Do not edit the generated outputs by hand.

My own environment was Python 3.14 with scikit-learn 1.9.0, which is the version that wrote the saved `.joblib` files. Loading them with a very different scikit-learn version may fail or warn.

## Saved models

| File | Made in | What it is |
|---|---|---|
| `titanic_survival.joblib` + `titanic_survival_card.json` | Lab 2 | Logistic-regression pipeline, best `C=0.1` with balanced class weights, final test ROC AUC 0.837 |
| `insurance_charges.joblib` | Lab 2 | Linear-regression pipeline for medical charges |
| `breast_cancer_svm.joblib` | Lab 3 | Tuned SVM |
| `digits_pca_logistic.joblib` | Lab 4 | PCA followed by logistic regression on handwritten digits |
| `gridworld_q_table.npy` | Lab 6 | Learned Q-table for the 4×4 grid world |

`all_experiments/predict_passenger.py` uses the saved Lab 2 Titanic model. Run it from inside `all_experiments/`:

```bash
cd all_experiments
python predict_passenger.py --pclass 1 --sex female --age 29 --fare 80
```

```
Predicted Class: 1 (Survived)
Survival Probability: 0.9184
```

Optional flags: `--sibsp`, `--parch`, `--embarked` (C, Q or S; default S).

## Author

Nur Farah binti Ahmad Nazri, Universiti Teknologi PETRONAS.
