# OpenSpec: Change Proposal 001

## 1. Proposal Title

Implement Core ML Pipeline and Streamlit Dashboard

## 2. Description

This proposal details the implementation of the complete end-to-end spam classification machine learning pipeline and the interactive Streamlit user interface as required for Homework 3.

## 3. Scope of Work

1.  **Data Pre-processing:** Load the dataset, clean the text (tokenization, remove stop words, remove punctuation), and replace entities like URLs and phone numbers with generic tokens (e.g., <URL>, <PHONE>).
2.  **Feature Engineering:** Convert the processed text into numerical features using TF-IDF Vectorization.
3.  **Model Training:** Train a **Logistic Regression** model (or another specified model like Naive Bayes/SVM) using the vectorized data.
4.  **Model Persistence:** Save the trained model and the TF-IDF vectorizer to the `models/` directory.
5.  **Evaluation:** Calculate and store performance metrics: Confusion Matrix, Precision, Recall, and F1-score.
6.  **Streamlit UI Development:** Create the interactive dashboard in `src/app.py` to display:
    * Data Overview (Class Distribution, Token Replacements).
    * Top Tokens by Class visualization.
    * Model Performance (Confusion Matrix, ROC Curve, Precision-Recall Curve).
    * Threshold Sweep table showing (precision/recall/f1) vs. threshold.
    * Live Inference section for real-time testing.

## 4. Technical Details & Implementation Plan

### 4.1. Data Pre-processing (src/data_prep.py - *to be created*)

* **Input:** `data/sms_spam_no_header.csv`
* **Process:** Use Python's `re` module for pattern matching (URLs, numbers) and NLTK for text normalization.

### 4.2. Model Pipeline (src/model_train.py - *to be created*)

* **Split:** Use `train_test_split` with `test_size=0.20` and `random_state=42`.
* **Vectorization:** Apply `TfidfVectorizer` to the text data.
* **Model:** Use `sklearn.linear_model.LogisticRegression`.
* **Saving:** Use `joblib` to save the trained model (`models/lr_model.joblib`) and the vectorizer (`models/vectorizer.joblib`).

### 4.3. Streamlit UI (src/app.py)

* **Inputs:** Use `st.sidebar.slider` for `Test size` (fixed at 0.20) and `Decision threshold` (slider from 0.0 to 1.0, default 0.50).
* **Visualizations:** Use `matplotlib` or `plotly` integrated with `st.pyplot()` or `st.plotly_chart()` to display the required plots.
* **Live Inference:** Implement a function that loads the saved model and vectorizer to predict new, user-entered text.

## 5. Success Criteria

* The ML pipeline executes without errors, and the model and vectorizer are saved successfully.
* The `src/app.py` file runs as a Streamlit application and includes all visualizations shown in the assignment images.
* The application adheres to the `project.md` conventions (e.g., using test size 0.20 and seed 42).