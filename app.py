# Streamlit App for ML Assignment 2
# This app loads the models I trained earlier and lets you test them
# Note: Models are already trained, this just loads and tests them!

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# setting up the page
st.set_page_config(
    page_title="ML Model Evaluation",
    page_icon="📊",
    layout="wide"
)

# main title
st.title("📊 Income Prediction - Model Comparison")
st.markdown("**ML Assignment 2** - Comparing 5 Classification Models")
st.markdown("---")

# Sidebar
st.sidebar.header("🎯 Choose Model")
st.sidebar.info("Select a model to see how it performs. The models were trained on 80% of the data.")

# Models I trained
available_models = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'K-Nearest Neighbors': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Support Vector Machine': 'model/svm.joblib'
}

selected_model = st.sidebar.selectbox(
    "Pick a model:",
    list(available_models.keys())
)

# quick descriptions of each model
model_info = {
    'Logistic Regression': "Simple linear model, works well for baseline. Pretty fast.",
    'Decision Tree': "Tree-based model, easy to understand. Sometimes overfits though.",
    'K-Nearest Neighbors': "Uses 5 nearest neighbors. Can be slow with large data.",
    'Naive Bayes': "Probability-based. Assumes features are independent (not always true).",
    'Support Vector Machine': "SVM with RBF kernel. Usually gives good accuracy but takes time."
}

st.sidebar.markdown(f"**About:**")
st.sidebar.info(model_info[selected_model])

st.sidebar.markdown("---")

# Upload section
st.sidebar.header("📁 Test Your Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file with test data",
    type=['csv'],
    help="Should have features + target column"
)

# download sample if needed
if os.path.exists('test_data.csv'):
    with open('test_data.csv', 'rb') as f:
        st.sidebar.download_button(
            label="📥 Get Sample CSV",
            data=f,
            file_name="sample_test_data.csv",
            mime="text/csv"
        )

# format info
with st.sidebar.expander("ℹ️ CSV Format"):
    st.write("""
    Your CSV needs:
    - All the feature columns
    - A 'target' column (0 or 1)
    
    Like this:
    ```
    age, workclass, education-num, ..., target
    39, 7, 13, ..., 0
    50, 6, 13, ..., 1
    ```
    You can download the sample above to see the format.
    """)

# main area
if uploaded_file is not None:
    try:
        # reading the file
        data = pd.read_csv(uploaded_file)
        
        st.success(f"✓ Loaded! Found {data.shape[0]} rows and {data.shape[1]} columns")
        
        # make sure target exists
        if 'target' not in data.columns:
            st.error("❌ Missing 'target' column! Please check your CSV.")
            st.stop()
        
        # quick preview
        with st.expander("👀 Data Preview (first 10 rows)"):
            st.dataframe(data.head(10))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Samples", len(data))
            with col2:
                st.metric("Features", len(data.columns) - 1)
        
        st.markdown("---")
        
        # split into X and y
        X_test = data.drop(columns=['target'])
        y_test = data['target']
        
        # get model path
        model_path = available_models[selected_model]
        
        if not os.path.exists(model_path):
            st.error(f"❌ Couldn't find model file: {model_path}")
            st.info("You need to run the notebook first to train models!")
            st.stop()
        
        # loading the model
        with st.spinner(f"Loading {selected_model}... this might take a sec"):
            model = joblib.load(model_path)
        
        st.success(f"✓ {selected_model} ready!")
        
        # predictions (just loading saved model, not training again)
        with st.spinner("Running predictions..."):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # calculating all metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # showing results
        st.subheader(f"📈 Results for {selected_model}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("AUC", f"{auc:.4f}")
        
        with col2:
            st.metric("Precision", f"{precision:.4f}")
            st.metric("Recall", f"{recall:.4f}")
        
        with col3:
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("MCC", f"{mcc:.4f}")
        
        st.markdown("---")
        
        # confusion matrix and report
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{selected_model}')
            st.pyplot(fig)
            
            # percentages to understand better
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            st.write("**Accuracy by class:**")
            st.write(f"- Class 0 (<=50K): {cm_percent[0,0]:.1f}% correct")
            st.write(f"- Class 1 (>50K): {cm_percent[1,1]:.1f}% correct")
        
        with col2:
            st.subheader("📋 Detailed Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))
        
        st.markdown("---")
        
        # comparing all models
        st.subheader("🏆 Model Comparison")
        
        # check if we have the comparison file
        if os.path.exists('model_results.csv'):
            results_df = pd.read_csv('model_results.csv')
            
            # highlighting the one we're looking at now
            def highlight_selected(row):
                if row['Model'] == selected_model:
                    return ['background-color: #ffffcc'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                results_df.style.apply(highlight_selected, axis=1).format({
                    'Accuracy': '{:.4f}',
                    'AUC': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1': '{:.4f}',
                    'MCC': '{:.4f}'
                })
            )
            
            st.caption(f"💡 Current model ({selected_model}) is highlighted")
            
            # bar charts for comparison
            st.subheader("📊 Visual Comparison")
            
            metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                # highlight current model in different color
                colors = ['#ff6b6b' if m == selected_model else '#4dabf7' 
                         for m in results_df['Model']]
                results_df.plot(x='Model', y=metric, kind='bar', ax=ax, 
                              legend=False, color=colors)
                ax.set_title(f'{metric}')
                ax.set_xlabel('')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.info("Couldn't find model_results.csv. Make sure you ran the notebook first!")
        
        # reminding that this doesn't retrain
        st.markdown("---")
        st.info("ℹ️ **Note:** This app just loads and tests the saved models. It doesn't train anything new!")
        
    except Exception as e:
        st.error(f"Oops, something went wrong: {e}")
        st.info("Double check your CSV format matches the expected structure.")

else:
    # instructions when no file is uploaded
    st.info("👈 Upload a test CSV file from the sidebar to get started")
    
    st.markdown("### How to Use This App")
    st.write("""
    1. Pick a model from the sidebar
    2. Upload your test data CSV (needs to have a 'target' column)
    3. Check out the results - metrics, confusion matrix, etc.
    4. Compare with other models
    
    You can download a sample CSV from the sidebar if needed!
    """)
    
    st.markdown("### About This Assignment")
    st.write("""
    For this assignment I implemented 5 different classification models:
    - Logistic Regression
    - Decision Tree 
    - K-Nearest Neighbors (KNN)
    - Naive Bayes
    - Support Vector Machine (SVM)
    
    Each model is evaluated using 6 metrics: Accuracy, AUC, Precision, Recall, F1, and MCC.
    
    The models were trained on the Adult Income dataset to predict if someone earns >50K/year or not.
    """)

# footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>ML Assignment 2 | M.Tech AIML</p>
</div>
""", unsafe_allow_html=True)
