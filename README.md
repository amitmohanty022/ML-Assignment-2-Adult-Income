# ML Assignment 2 - Income Prediction
## Comparing 5 Classification Models

This project is for my ML assignment where I need to train 5 different classification models and compare them. I'm using the Adult Income dataset to predict whether someone makes more than $50K per year or not.

### What I Did

- Trained 5 models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM)
- Used sklearn Pipelines so everything stays clean and there's no data leakage
- Evaluated each model with 6 different metrics
- Saved all the trained models
- Built a Streamlit app to test them out

**Goal:** Figure out which model works best for this income prediction problem.

## About the Dataset

**Dataset:** Adult Income Dataset (also called Census Income)

**Where I got it:** Downloaded from OpenML, it's originally from the UCI repository

**Size and stuff:**
- 48,842 samples (pretty big dataset!)
- 14 features total
- Target is binary: either income ≤50K (label=0) or >50K (label=1)
- Has both numerical and categorical features

**Features in the data:**

Numerical ones (6):
- age
- fnlwgt (some census weight thing)
- education-num (years of education)
- capital-gain
- capital-loss  
- hours-per-week

Categorical ones (8):
- workclass (Private, Government, etc.)
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

**Target:**
- 0 = earns ≤50K/year
- 1 = earns >50K/year

**Class balance:**
- About 75% are in the ≤50K class
- About 25% in the >50K class
- So it's a bit imbalanced

**Data issues I dealt with:**
- Some missing values (filled them with median/mode)
- Had to convert categorical features to numbers
- All this is handled in the code automatically

## Project Structure

```
ML-Assignment-2/
│
├── data/
│   └── adult_income.csv           # the dataset
│
├── model/                          # where trained models are saved
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
└── .gitignore                      
```

## Links

**GitHub:** https://github.com/amitmohanty022/ML-Assignment-2-Adult-Income

**Live App:** https://ml-assignment-2-adult-income-ywkff77qrwyslvgxp4pkf6.streamlit.app/

## The 5 Models I Used

For this assignment I had to implement 5 classification algorithms:

1. **Logistic Regression** - Basic linear classifier, good starting point
2. **Decision Tree** - Tree-based model, easy to visualize 
3. **K-Nearest Neighbors (KNN)** - Set k=5, instance-based learning
4. **Naive Bayes** - Gaussian NB, fast and simple
5. **Support Vector Machine (SVM)** - Used RBF kernel, usually gives good results

### Metrics (6 total as required):
1. Accuracy - overall correctness
2. AUC Score - area under ROC curve
3. Precision - positive prediction accuracy
4. Recall - how many positives we caught
5. F1 Score - harmonic mean of precision and recall
6. MCC - Matthews Correlation Coefficient

### What I Observed

| Model | My Thoughts |
|-------|-------------|
| **Logistic Regression** | Simple but effective. Works well when features are somewhat linearly separable. Good interpretability - you can see which features matter most. Expected this to be a solid baseline. |
| **Decision Tree** | Really good with mixed data types. Can handle categorical stuff easily. The tree rules are easy to understand. Main issue is it might overfit - had to watch the depth. |
| **KNN** | This one was interesting. Distance-based so scaling matters a lot. With k=5 it should balance bias-variance reasonably. Might struggle with high dimensionality though. |
| **Naive Bayes** | Super fast to train! The independence assumption isn't perfect for this data but it often works anyway. Good for a quick baseline. |
| **Support Vector Machine** | Took the longest to train but gave good results. RBF kernel helps capture non-linear patterns. Needs scaling (which is why I used Pipeline). |
| **Best Model** | Based on my results, **SVM performed best** with 85.26% accuracy and 89.40% AUC. But Logistic Regression and KNN were also pretty good. Decision Tree had decent F1 score. |

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

Or install them one by one:
```bash
pip3 install jupyter pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook ML_Assignment2.ipynb
```

This will:
- Load and explore the data
- Train all 5 models (using Pipeline architecture)
- Calculate the 6 metrics for each
- Save the trained models as .joblib files
- Create test_data.csv for the app

### 4. Run the Streamlit App
```bash
python3 -m streamlit run app.py
```

or just:
```bash
streamlit run app.py
```
```bash

Opens at `http://localhost:8501`

Then upload the test_data.csv and try out different models!

## What the App Does

The Streamlit app is basically a testing interface for the models I trained. You can:
- Pick any model from the dropdown
- Upload a CSV with test data  
- See all the metrics (accuracy, AUC, etc.)
- Check out the confusion matrix
- Compare all 5 models at once

Just make sure your CSV has all the features plus a 'target' column.

## Deployment

I deployed this on Streamlit Cloud so it's accessible online.

**Live URL:** https://ml-assignment-2-adult-income-ywkff77qrwyslvgxp4pkf6.streamlit.app/

To deploy yours:
1. Push code to GitHub
2. Go to streamlit.io/cloud
3. Sign in with GitHub
4. Click "New app"
5. Pick your repo, select app.py
6. Hit Deploy!

Takes about 5 minutes. The requirements.txt file tells Streamlit what packages to install.

## Quick Metrics Guide

For anyone not familiar with these metrics:

- **Accuracy** - Simple: % of correct predictions out of total
- **AUC** - How well the model distinguishes between the two classes (0 to 1, higher is better)
- **Precision** - When model predicts positive, how often is it right?
- **Recall** - Of all actual positives, how many did we find?
- **F1 Score** - Combines precision and recall (good when classes are imbalanced)
- **MCC** - Correlation between predictions and reality (-1 to +1, accounts for class imbalance)

## My Results

After running everything:

| Model | Accuracy | AUC | F1 Score |
|-------|----------|-----|----------|
| Logistic Regression | 0.8302 | 0.8563 | 0.5601 |
| Decision Tree | 0.8120 | 0.7461 | 0.6120 |
| K-Nearest Neighbors | 0.8328 | 0.8525 | 0.6284 |
| Naive Bayes | 0.8050 | 0.8564 | 0.4556 |
| **Support Vector Machine** | **0.8526** | **0.8940** | **0.6455** |

**Winner:** SVM got the best scores overall! Though all models did reasonably well (80%+ accuracy).

## Things I Noticed

- The dataset has way more people earning ≤50K than >50K (class imbalance)
- Features like education, occupation, and hours-per-week seem important
- SVM took longest to train but gave best results
- Decision Tree was fastest and still pretty good
- Had to handle missing values (some fields had '?' instead of actual values)
- Feature scaling really matters for SVM and KNN

## Requirements

See `requirements.txt` for full list. Main ones:
- Python 3.7+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- streamlit
- joblib

## Files Explained

- **ML_Assignment2.ipynb** - Main notebook with all the work
- **app.py** - Streamlit app code
- **requirements.txt** - Package versions
- **model/** folder - All 5 trained models saved here
- **test_data.csv** - Test set for the app
- **model_results.csv** - Metrics comparison table
- **full_dataset.csv** - Complete Adult Income dataset

---

Made for ML Assignment 2 | M.Tech AIML | November 2026
