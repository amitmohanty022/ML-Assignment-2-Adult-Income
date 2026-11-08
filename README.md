# Machine Learning Assignment 2
## Classification Model Comparison with Pipeline Architecture

### Problem Statement

This project implements and compares **5 classification algorithms** on the **Adult Income Dataset** to predict whether an individual's annual income exceeds $50,000. The project follows ML best practices including:

- Proper train-test split to avoid data leakage
- sklearn Pipeline architecture combining preprocessing and models
- Comprehensive evaluation with 6 metrics
- Saved pipeline deployment (no retraining in production)
- Interactive Streamlit application for model evaluation

**Goal:** Compare model performance and identify the best classifier for income prediction based on demographic and employment features.

## Dataset Description

**Dataset:** Adult Income Dataset (Census Income)

**Source:** I used the dataset from OpenML (originally from UCI ML Repository via scikit-learn)

**Basic Info:**
- Total samples: 48,842 
- Number of features: 14
- Target: Binary (Income ≤50K = 0, Income >50K = 1)
- Mix of categorical and numerical features

**Features:**
The dataset contains demographic and employment information:

**Numerical Features (6):**
- age - Age of the person
- fnlwgt - Final weight (census weighting)
- education-num - Years of education
- capital-gain - Capital gains
- capital-loss - Capital losses
- hours-per-week - Working hours per week

**Categorical Features (8):**
- workclass - Type of employment (Private, Government, etc.)
- education - Highest level of education
- marital-status - Marital status
- occupation - Type of occupation
- relationship - Family relationship
- race - Race
- sex - Gender
- native-country - Country of origin

**Target Variable:**
- 0 = Income ≤50K per year
- 1 = Income >50K per year

**Class Distribution:**
- Income ≤50K: ~75% (majority class)
- Income >50K: ~25% (minority class)

**Data Quality:**
- Some missing values (handled by filling with median/mode)
- Categorical features converted to numeric codes
- All preprocessing done automatically in the code

## Repository Structure

```
ML-Assignment-2/
│
├── data/
│   └── adult_income.csv           # Full dataset
│
├── model/                          # Saved trained pipelines
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── ML_Assignment2.ipynb            # Main Jupyter notebook
├── app.py                          # Streamlit evaluation app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── test_data.csv                   # Test set with labels
├── model_results.csv               # Metrics comparison table
└── .gitignore                      # Git ignore rules
```

## GitHub Repository Link

[Your GitHub Repository URL - Replace with actual URL]

## Live Streamlit Application

[Your Streamlit App URL - Replace with actual URL]

## Models Used

I implemented 5 classification algorithms as required:

1. **Logistic Regression** - Simple linear model
2. **Decision Tree** - Tree-based approach
3. **K-Nearest Neighbors (KNN)** - Used k=5 neighbors
4. **Naive Bayes** - Gaussian NB for continuous features
5. **Support Vector Machine (SVM)** - With RBF kernel

### For each model, I calculated these 6 metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. MCC (Matthews Correlation Coefficient)

*Note: These will be generated when you run train_models.py - values shown below are from my run*

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Should perform well on this dataset since it handles both numerical and categorical features. Good for interpretability - can see which factors most influence income predictions. Expected accuracy around 80-85%. |
| **Decision Tree** | Works great with mixed feature types and can capture complex relationships between education, occupation, and income. Easy to interpret the decision rules. Might overfit if not pruned properly. |
| **KNN** | Performance might vary since we have both categorical and numerical features. The distance calculation could be affected by feature scaling. Used k=5 neighbors as a starting point. |
| **Naive Bayes** | Despite the independence assumption, should work reasonably well. Fast training is a plus for this larger dataset. Good baseline model to compare against. |
| **Support Vector Machine** | Expected to perform well after feature scaling. Might take longer to train with 48K samples but should give good accuracy. RBF kernel can capture non-linear relationships between features. |
| **Overall Winner for your dataset** | Results will vary, but typically **Logistic Regression** or **SVM** perform best on this dataset. Decision Trees also work well due to the mix of categorical features. The winner will be determined after training - check the comparison table! |

## How to Run This Project

### 1. Clone Repository
```bash
git clone [your-github-repo-url]
cd ML-Assignment-2
```

### 2. Install Dependencies
```bash
pip3 install -r requirements.txt
```

Or install individually:
```bash
pip3 install jupyter pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook ML_Assignment2.ipynb
```

Execute all cells to:
- Load and explore data
- Train all 5 models with Pipelines
- Evaluate with 6 metrics
- Save trained pipelines
- Generate test_data.csv

### 4. Run Streamlit App
```bash
python3 -m streamlit run app.py
```

Or simply:
```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

### 5. Evaluate Models
- Upload `test_data.csv` in the app
- Select different models from dropdown
- View metrics, confusion matrix, and comparisons

**Note:** The app loads saved pipelines and only performs evaluation - no retraining occurs.

## Using the App

### Training

When you run `python3 train_models.py`, it will:
- Download the Adult Income dataset (48,842 samples)
- Process categorical features
- Split into 80% train and 20% test
- Train all 5 models
- Save everything in the model/ folder
- Create test_data.csv and model_results.csv

**Note:** First run may take 2-3 minutes to download the dataset.
- Save everything in the model/ folder
- Create test_data.csv and model_results.csv

### Web App Features

The Streamlit app lets you:
- Pick any of the 5 models from dropdown
- Upload your own test data (CSV format)
- See all 6 metrics for the selected model
- View confusion matrix
- See classification report
- Compare all models side-by-side

The test CSV should have all 30 features plus a 'target' column.

## Deployment on Streamlit Cloud

### Steps I followed:

1. Pushed all code to GitHub
2. Went to https://streamlit.io/cloud
3. Logged in with GitHub
4. Created new app
5. Selected repository, branch (main), and app.py
6. Clicked Deploy

The deployment takes a few minutes. Once done, you get a public URL that anyone can access.

**Note:** Make sure requirements.txt has all dependencies and the model files are in the repo.

## Metrics Explained

- **Accuracy:** % of correct predictions
- **AUC:** Area under ROC curve - how well model separates classes
- **Precision:** Of positive predictions, how many were actually positive
- **Recall:** Of actual positives, how many did we catch
- **F1 Score:** Harmonic mean of precision and recall
- **MCC:** Correlation coefficient between predicted and actual (-1 to +1)

## What I Learned

- Handling categorical features is important - converted them to numeric codes
- Feature scaling matters for SVM and KNN
- Larger datasets (48K samples) take longer to train but give more robust results
- Census data has class imbalance (more people earn ≤50K)
- Different models handle mixed feature types differently
- Real-world data requires preprocessing (missing values, categorical encoding)

## Files in This Repo

```
├── app.py                    # Streamlit web app
├── train_models.py           # Script to train all models
├── requirements.txt          # Python packages needed
├── README.md                 # This file
├── test_data.csv            # Test set (created after training)
├── full_dataset.csv         # Complete dataset
├── model_results.csv        # All metrics for comparison
└── model/                   # Folder with trained models
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── scaler.joblib
```

---

**Created for:** BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2  
**Date:** November 2026
