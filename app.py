# Streamlit App for ML Assignment 2
# Loads saved pipelines and evaluates uploaded test data
# No retraining - only evaluation!

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

# Page config
st.set_page_config(
    page_title="ML Model Evaluation",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 ML Classification Model Evaluation")
st.markdown("**Course:** M.Tech (AIML) - Machine Learning Assignment 2")
st.markdown("---")

# Sidebar
st.sidebar.header("🎯 Model Selection")
st.sidebar.info("This app evaluates pre-trained models on uploaded test data. No retraining occurs.")

# Available models
available_models = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'KNN': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Support Vector Machine': 'model/svm.joblib'
}

selected_model = st.sidebar.selectbox(
    "Choose a model to evaluate:",
    list(available_models.keys())
)

# Model descriptions
model_descriptions = {
    'Logistic Regression': "Linear model for binary classification. Fast and interpretable.",
    'Decision Tree': "Tree-based model using recursive partitioning. Highly interpretable.",
    'KNN': "Instance-based learning using 5 nearest neighbors.",
    'Naive Bayes': "Probabilistic classifier based on Bayes theorem.",
    'Support Vector Machine': "Maximum margin classifier with RBF kernel."
}

st.sidebar.markdown(f"**About this model:**")
st.sidebar.info(model_descriptions[selected_model])

st.sidebar.markdown("---")

# File upload
st.sidebar.header("📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload test data with all features and 'target' column"
)

# Sample data download
if os.path.exists('test_data.csv'):
    with open('test_data.csv', 'rb') as f:
        st.sidebar.download_button(
            label="📥 Download Sample CSV",
            data=f,
            file_name="sample_test_data.csv",
            mime="text/csv"
        )

# Expected format info
with st.sidebar.expander("ℹ️ Expected CSV Format"):
    st.write("""
    Your CSV should include:
    - All feature columns (numerical, encoded)
    - A 'target' column with values: 0 or 1
    
    Example structure:
    ```
    age, workclass, education-num, ..., target
    39, 7, 13, ..., 0
    50, 6, 13, ..., 1
    ```
    """)

# Main content
if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        
        st.success(f"✓ Data loaded successfully! Shape: {data.shape}")
        
        # Check for target column
        if 'target' not in data.columns:
            st.error("❌ CSV must contain a 'target' column!")
            st.stop()
        
        # Show data preview
        with st.expander("👀 Preview Uploaded Data"):
            st.dataframe(data.head(10))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Samples", len(data))
            with col2:
                st.metric("Features", len(data.columns) - 1)
        
        st.markdown("---")
        
        # Separate features and target
        X_test = data.drop(columns=['target'])
        y_test = data['target']
        
        # Load selected model
        model_path = available_models[selected_model]
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please run the Jupyter notebook first to train and save models.")
            st.stop()
        
        # Load model
        with st.spinner(f"Loading {selected_model}..."):
            model = joblib.load(model_path)
        
        st.success(f"✓ {selected_model} loaded successfully!")
        
        # Make predictions (NO TRAINING!)
        with st.spinner("Evaluating model..."):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Display metrics
        st.subheader(f"📈 {selected_model} - Evaluation Results")
        
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
        
        # Confusion Matrix and Classification Report
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'{selected_model}')
            st.pyplot(fig)
            
            # Show percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            st.write("**Row Percentages:**")
            st.write(f"- Class 0: {cm_percent[0,0]:.1f}% correct, {cm_percent[0,1]:.1f}% wrong")
            st.write(f"- Class 1: {cm_percent[1,1]:.1f}% correct, {cm_percent[1,0]:.1f}% wrong")
        
        with col2:
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))
        
        st.markdown("---")
        
        # Compare all models if results file exists
        st.subheader("🏆 Compare All Models")
        
        if os.path.exists('model_results.csv'):
            results_df = pd.read_csv('model_results.csv')
            
            # Highlight current model
            def highlight_selected(row):
                if row['Model'] == selected_model:
                    return ['background-color: lightyellow'] * len(row)
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
            
            st.caption(f"💡 {selected_model} is highlighted")
            
            # Visual comparison
            st.subheader("📊 Performance Comparison")
            
            metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx]
                colors = ['coral' if m == selected_model else 'steelblue' 
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
            st.info("model_results.csv not found. Run the notebook to generate comparison data.")
        
        # Note about no retraining
        st.markdown("---")
        st.info("⚠️ **Note:** This app only EVALUATES pre-trained models. No retraining occurs when you upload data.")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please check that your CSV format matches the expected structure.")

else:
    st.info("👈 Please upload a test CSV file from the sidebar to begin evaluation.")
    
    st.markdown("### 📖 How to Use")
    st.write("""
    1. **Select a model** from the sidebar dropdown
    2. **Upload test data** CSV file (must include 'target' column)
    3. **View results:** metrics, confusion matrix, and classification report
    4. **Compare models** in the comparison section
    
    **Sample data available:** Click "Download Sample CSV" in sidebar
    """)
    
    st.markdown("### 🎯 About This Project")
    st.write("""
    This application demonstrates:
    - **5 Classification Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM
    - **6 Evaluation Metrics:** Accuracy, AUC, Precision, Recall, F1, MCC
    - **Pipeline Architecture:** Preprocessing and model saved together
    - **No Retraining:** Models are pre-trained and only evaluated here
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>BITS Pilani | M.Tech (AIML) | Machine Learning Assignment 2</p>
</div>
""", unsafe_allow_html=True)
