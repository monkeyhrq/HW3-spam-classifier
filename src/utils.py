import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
import string

# --- 1. 資料清理函式 ---
def clean_text(text):
    """清理文字：移除標點符號、數字、並轉換為小寫。"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 2. 資料載入與前處理 ---
def load_and_preprocess_data(data_path, text_col='col_1', label_col='col_0', test_size=0.2, random_state=42):
    """載入資料、清理、切割訓練/測試集並進行向量化。"""
    
    try:
        df = pd.read_csv(data_path, encoding='latin-1', header=None, names=[label_col, text_col])
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {data_path}。")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"載入檔案時發生錯誤: {e}")
        return None, None, None, None, None, None
    
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df['label'] = df[label_col].map({'ham': 0, 'spam': 1})
    
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    vectorizer = TfidfVectorizer(max_features=5000) 
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    X_test_raw = X_test
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer, X_test_raw

# --- 3. 模型訓練與儲存 ---
def train_and_save_model(X_train_vec, y_train, model_type='LogisticRegression', model_path='models/lr_model.joblib'):
    """訓練分類模型並將模型儲存。"""
    
    if model_type == 'LogisticRegression':
        model = LogisticRegression(solver='liblinear', random_state=42)
    elif model_type == 'NaiveBayes':
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
    else:
        raise ValueError(f"不支援的模型類型: {model_type}")

    model.fit(X_train_vec, y_train)
    joblib.dump(model, model_path)
    
    return model

# --- 4. 模型評估 ---
def evaluate_model(model, X_test_vec, y_test, threshold=0.5):
    """評估模型，並根據閾值計算指標。"""
    
    y_proba = model.predict_proba(X_test_vec)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_proba)

    metrics = {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_proba': y_proba,
        'y_test': y_test,
        'fpr': fpr,
        'tpr': tpr,
        'precision_vals': precision_vals,
        'recall_vals': recall_vals
    }
    
    return metrics

# --- 5. 實時預測 ---
def predict_message(message, vectorizer, model, threshold=0.5):
    """對單個訊息進行預測。"""
    
    cleaned_msg = clean_text(message)
    try:
        msg_vec = vectorizer.transform([cleaned_msg])
    except AttributeError:
        return "Model/Vectorizer Error", 0.0

    proba = model.predict_proba(msg_vec)[0, 1]
    prediction = 'spam' if proba >= threshold else 'ham'
    
    return prediction, proba

# --- 6. 詞彙分析函式 ---
def get_top_tokens(vectorizer, X_train_vec, y_train, top_n=20):
    """計算並返回每個類別 (Ham/Spam) 中詞彙的 TF-IDF 總和排名。"""
    
    feature_names = vectorizer.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_vec.toarray(), columns=feature_names)
    X_train_df['label'] = y_train.values 

    spam_df = X_train_df[X_train_df['label'] == 1].drop(columns=['label'])
    ham_df = X_train_df[X_train_df['label'] == 0].drop(columns=['label'])
    
    spam_sums = spam_df.sum().sort_values(ascending=False).head(top_n)
    ham_sums = ham_df.sum().sort_values(ascending=False).head(top_n)
    
    top_tokens_ham = pd.DataFrame({'token': ham_sums.index, 'frequency': ham_sums.values}).sort_values(by='frequency', ascending=True)
    top_tokens_spam = pd.DataFrame({'token': spam_sums.index, 'frequency': spam_sums.values}).sort_values(by='frequency', ascending=True)
    
    return top_tokens_ham, top_tokens_spam