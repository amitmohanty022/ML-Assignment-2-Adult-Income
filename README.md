# ML Assignment 2 - Income Prediction
## Comparing 5 Classification Models

## a. Problem statement

The goal of this project is to predict whether a person earns more than $50K per year using census information such as age, education, occupation, marital status and hours worked per week. I trained 5 classification models on the same dataset, compared them using 6 evaluation metrics, and built a Streamlit app that loads the saved models and evaluates them on uploaded test data without retraining.

## b. Dataset description

- **Dataset:** Adult Income Dataset (also called Census Income)
- **Source:** Downloaded from OpenML, originally from the UCI Machine Learning Repository - <PASTE THE DATASET URL YOU USED>
- **Instances:** 48,842
- **Input features:** 14 (6 numerical, 8 categorical)
- **Target:** binary
  - 0 = earns ≤50K/year
  - 1 = earns >50K/year

**Numerical features (6):**
- age
- fnlwgt (census sampling weight)
- education-num (years of education)
- capital-gain
- capital-loss
- hours-per-week

**Categorical features (8):**
- workclass (Private, Government, etc.)
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

**Class balance:** about 76% of people are in the ≤50K class and 24% in the >50K class, so the dataset is imbalanced. Because of this, I used MCC and F1 alongside accuracy to compare the models.

**Data issues and preprocessing:**
- Some fields had '?' instead of an actual value. These were treated as missing.
- Missing numerical values were filled with the median and missing categorical values with the most frequent value.
- Categorical features were one-hot encoded and numerical features were scaled.
- The data was split 80/20 with stratification (39,073 train rows, 9,769 test rows), so both parts have the same class ratio.
- All preprocessing is done inside sklearn Pipelines and fitted only on the training data, so there is no data leakage.

## c. GitHub repository

**GitHub:** https://github.com/amitmohanty022/ML-Assignment-2-Adult-Income

**Live App:** https://ml-assignment-2-adult-income-ywkff77qrwyslvgxp4pkf6.streamlit.app/

## d. Models used

1. **Logistic Regression** - linear classifier, used as the baseline
2. **Decision Tree** - tree-based model, easy to interpret
3. **K-Nearest Neighbors (KNN)** - k=5, distance-based
4. **Naive Bayes** - Gaussian NB
5. **Support Vector Machine (SVM)** - RBF kernel

Each model is a Pipeline (preprocessing + classifier) trained on the same training split and evaluated on the same test split.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8302 | 0.8563 | 0.7369 | 0.4517 | 0.5601 | 0.4835 |
| Decision Tree | 0.8120 | 0.7461 | 0.6045 | 0.6198 | 0.6120 | 0.4880 |
| KNN | 0.8328 | 0.8525 | 0.6714 | 0.5907 | 0.6284 | 0.5229 |
| Naive Bayes | 0.8050 | 0.8564 | 0.6865 | 0.3409 | 0.4556 | 0.3849 |
| Support Vector Machine | 0.8526 | 0.8940 | 0.7604 | 0.5607 | 0.6455 | 0.5654 |

All metrics are on the held-out 20% test set (9,769 rows). Precision, Recall and F1 are for the positive class (income >50K).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | WRITE YOUR OWN OBSERVATION HERE |
| Decision Tree | WRITE YOUR OWN OBSERVATION HERE |
| KNN | WRITE YOUR OWN OBSERVATION HERE |
| Naive Bayes | WRITE YOUR OWN OBSERVATION HERE |
| Support Vector Machine | WRITE YOUR OWN OBSERVATION HERE |
| Overall, winner for your dataset? | WRITE YOUR OWN ANSWER HERE |

## Project Structure

```
ML-Assignment-2-Adult-Income/
│
├── data/
│   └── adult_income.csv           # the dataset
│
├── model/                          # trained pipelines
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── ML_Assignment2.ipynb            # main notebook with all the code
├── app.py                          # streamlit app for testing models
├── requirements.txt                # packages needed
├── README.md                       # this file
├── test_data.csv                   # test set for evaluation
├── model_results.csv               # comparison of all models
├── full_dataset.csv                # complete dataset for reference
└── .gitignore
```

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/amitmohanty022/ML-Assignment-2-Adult-Income.git
cd ML-Assignment-2-Adult-Income
```

### 2. Install packages
```bash
pip3 install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook ML_Assignment2.ipynb
```

This will:
- Load and explore the data
- Train all 5 models (using Pipelines)
- Calculate the 6 metrics for each
- Save the trained models as .joblib files
- Create test_data.csv for the app

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload test_data.csv and try the different models. The app loads the saved models from model/ and does not retrain them.

## What the App Does

The Streamlit app is a testing interface for the trained models. You can:
- Pick any model from the dropdown
- Upload a CSV with test data
- See all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Check the confusion matrix and classification report
- Compare all 5 models at once

The CSV must have all the feature columns plus the 'target' column.

## Deployment

Deployed on Streamlit Community Cloud:

**Live URL:** https://ml-assignment-2-adult-income-ywkff77qrwyslvgxp4pkf6.streamlit.app/

Steps used:
1. Push code to GitHub
2. Go to streamlit.io/cloud and sign in with GitHub
3. Click "New app"
4. Pick the repo, branch main, and app.py
5. Click Deploy

## Quick Metrics Guide

- **Accuracy** - % of correct predictions out of total
- **AUC** - how well the model separates the two classes (0 to 1, higher is better)
- **Precision** - when the model predicts >50K, how often it is right
- **Recall** - of all people who actually earn >50K, how many the model found
- **F1 Score** - combines precision and recall (useful when classes are imbalanced)
- **MCC** - correlation between predictions and actual labels (-1 to +1, accounts for class imbalance)

## Files Explained

- **ML_Assignment2.ipynb** - main notebook with all the work
- **app.py** - Streamlit app code
- **requirements.txt** - package versions
- **model/** - the 5 trained pipelines
- **test_data.csv** - test set for the app
- **model_results.csv** - metrics comparison table
- **full_dataset.csv** - complete Adult Income dataset

---

Made for ML Assignment 2 | M.Tech AIML | November 2026
