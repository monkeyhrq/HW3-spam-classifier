import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os 
from pathlib import Path 
from utils import load_and_preprocess_data, train_and_save_model, evaluate_model, predict_message, get_top_tokens
from sklearn.metrics import precision_score, recall_score, f1_score # 為了閾值掃描新增

# --- 設定路徑和常數 ---

# 獲取 app.py 腳本的絕對路徑
current_file_path = Path(__file__).absolute() 

# 向上兩層到達 HW3 根目錄 (假設 app.py 在 src/ 內)
BASE_DIR = current_file_path.parent.parent 

# 使用 Path 構造數據集路徑
DATA_PATH = BASE_DIR / 'Chapter03' / 'datasets' / 'sms_spam_no_header.csv' 
MODEL_PATH = BASE_DIR / 'models' / 'lr_model.joblib'
VECTORIZER_PATH = BASE_DIR / 'models' / 'vectorizer.joblib'

TEST_SPAM_MSG = "Congratulations! You have won $1000 cash. Claim your prize now! Text back 'FREE' to 8888"
TEST_HAM_MSG = "Hey, just finished the meeting. Can we review the project notes at 4pm?"

# --- 快取函式 ---

@st.cache_resource
def load_and_cache_data_pipeline(test_size, seed):
    """載入、前處理並切割數據集，快取結果。"""
    
    # 將 Path 物件轉換為字串傳遞給 pandas
    data_path_str = str(DATA_PATH)
    
    X_train_vec, X_test_vec, y_train, y_test, vectorizer, X_test_raw = \
        load_and_preprocess_data(
            data_path_str, 
            test_size=test_size, 
            random_state=seed
        )
    if X_train_vec is None:
         # 數據載入失敗的錯誤訊息已經在 utils.py 中處理，這裡只返回 None
         return None, None, None, None, None, None
         
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer, X_test_raw

@st.cache_resource
def load_model_and_vectorizer():
    """載入模型和向量化工具。"""
    # 確保 models 目錄存在
    os.makedirs(str(BASE_DIR / 'models'), exist_ok=True) 
    
    model_path_str = str(MODEL_PATH)
    vectorizer_path_str = str(VECTORIZER_PATH)
    
    if not os.path.exists(model_path_str) or not os.path.exists(vectorizer_path_str):
        return None, None
    
    try:
        model = joblib.load(model_path_str)
        vectorizer = joblib.load(vectorizer_path_str)
        return model, vectorizer
    except Exception as e:
        # st.error(f"載入模型或向量化工具失敗: {e}") # 避免載入失敗時一直彈出錯誤
        return None, None

# --- Streamlit 應用程式主體 ---

st.set_page_config(layout="wide", page_title="Spam/Ham 分類器儀表板")

st.title("📧 Spam/Ham Classifier — Phase 4 Visualizations")
st.subheader("互動式數據分佈、特徵和模型性能儀表板")

# -----------------------------------------------------
# 側邊欄：輸入控制項 
# -----------------------------------------------------
st.sidebar.header("Inputs (輸入控制項)")

st.sidebar.selectbox("Dataset CSV (數據集)", [str(DATA_PATH)]) 
st.sidebar.selectbox("Label column (標籤欄位)", ["col_0"])
st.sidebar.selectbox("Text column (文本欄位)", ["col_1"])

st.sidebar.text_input("Models dir", "models")

test_size = st.sidebar.slider("Test size (測試集大小)", 0.05, 0.50, 0.20, 0.05)
seed = st.sidebar.number_input("Seed (隨機種子)", 0, 100, 42)
decision_threshold = st.sidebar.slider("Decision threshold (決策閾值)", 0.0, 1.0, 0.50, 0.05)


# -----------------------------------------------------
# 主面板：載入數據和模型
# -----------------------------------------------------

# 初始載入數據和模型
X_train_vec, X_test_vec, y_train, y_test, vectorizer, X_test_raw = \
    load_and_cache_data_pipeline(test_size, seed)

model, vectorizer_loaded = load_model_and_vectorizer()


# --- 重新訓練模型按鈕 (修正快取衝突的邏輯) ---
if st.button("重新訓練模型 (Logistic Regression)"):
    
    # 步驟 1: 強制清除所有相關快取，確保訓練時獲取最新的 Vectorizer
    load_and_cache_data_pipeline.clear() 
    load_model_and_vectorizer.clear()
    
    # 重新運行數據管道以確保我們拿到最新的 X_train_vec 和 vectorizer
    X_train_vec, X_test_vec, y_train, y_test, vectorizer, X_test_raw = \
        load_and_cache_data_pipeline(test_size, seed)
    
    if X_train_vec is not None and y_train is not None:
        try:
            # 步驟 2: 訓練模型並儲存
            trained_model = train_and_save_model(X_train_vec, y_train, model_type='LogisticRegression', model_path=str(MODEL_PATH))
            joblib.dump(vectorizer, str(VECTORIZER_PATH)) # 儲存 vectorizer
            
            st.success("模型和向量化工具已重新訓練並儲存！")
            st.rerun() # 重新運行以載入新模型
        except Exception as e:
            st.error(f"模型訓練失敗: {e}")
    else:
        st.error("訓練失敗：無法載入數據，請檢查數據集檔案。")


# 檢查數據是否可用 
if X_train_vec is None:
    st.error("無法顯示數據概覽，請檢查數據集檔案。")
    st.stop()


# -----------------------------------------------------
# 數據概覽 (Data Overview) 
# -----------------------------------------------------
st.header("Data Overview (數據概覽)")

try:
    original_df = pd.read_csv(str(DATA_PATH), encoding='latin-1', header=None, names=['col_0', 'col_1'])
    class_counts = original_df['col_0'].value_counts()
except Exception:
    st.error("載入數據集失敗。")
    st.stop() 

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Class distribution (類別分佈)")
    fig_dist = go.Figure(data=[go.Bar(x=class_counts.index, y=class_counts.values)])
    fig_dist.update_layout(xaxis_title="類別", yaxis_title="計數", height=400)
    st.plotly_chart(fig_dist, use_container_width=True) 

with col2:
    st.subheader("Token replacements in cleaned text (近似)")
    token_replacements = pd.DataFrame({
        'Token': ['<URL>', '<EMAIL>', '<PHONE>', '<NUM>'],
        'Count': ['未實現', '未實現', '未實現', '已移除'] 
    })
    st.dataframe(token_replacements, hide_index=True)


st.markdown("---")
# -----------------------------------------------------
# 模型效能指標 
# -----------------------------------------------------

st.header("Model Performance (模型性能)")

if model is not None and vectorizer_loaded is not None and X_test_vec is not None:
    try:
        # 運行評估
        metrics = evaluate_model(model, X_test_vec, y_test, threshold=decision_threshold)
        cm = metrics['confusion_matrix']

        st.subheader("Model Performance (Test)")
        col3, col4 = st.columns([1, 1])

        with col3:
            st.markdown("##### Confusion Matrix")
            cm_df = pd.DataFrame(cm, 
                                index=['true_0 (ham)', 'true_1 (spam)'], 
                                columns=['pred_0 (ham)', 'pred_1 (spam)'])
            st.dataframe(cm_df)
            
            st.markdown(f"**Precision (精確度):** `{metrics['precision']:.4f}`")
            st.markdown(f"**Recall (召回率):** `{metrics['recall']:.4f}`")
            st.markdown(f"**F1 Score (F1 分數):** `{metrics['f1']:.4f}`")


        with col4:
            # --- ROC 和 Precision-Recall 曲線 ---
            st.subheader("ROC & Precision-Recall Curves")
            
            fig_curves = make_subplots(rows=1, cols=2, subplot_titles=("ROC", "Precision-Recall"))
            
            # ROC 曲線
            fig_curves.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'], mode='lines', name='ROC Curve'), row=1, col=1)
            fig_curves.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'), row=1, col=1)
            fig_curves.update_xaxes(title_text="FPR (假陽性率)", row=1, col=1)
            fig_curves.update_yaxes(title_text="TPR (真陽性率)", row=1, col=1)

            # Precision-Recall 曲線
            fig_curves.add_trace(go.Scatter(x=metrics['recall_vals'], y=metrics['precision_vals'], mode='lines', name='PR Curve'), row=1, col=2)
            fig_curves.update_xaxes(title_text="Recall (召回率)", row=1, col=2)
            fig_curves.update_yaxes(title_text="Precision (精確度)", row=1, col=2)

            fig_curves.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_curves, use_container_width=True)

    except ValueError as ve:
        st.error(f"模型評估失敗: {ve}")
        st.warning("可能是模型與特徵數不匹配。請點擊上方的 '重新訓練模型' 按鈕。")
    except Exception as e:
        st.error(f"模型評估時發生未知錯誤: {e}")

else:
    st.info("模型性能區塊：請先成功訓練和載入模型後，此處才會顯示性能指標和圖表。")

st.markdown("---")
# -----------------------------------------------------
# 閾值掃描表格 (Threshold Sweep Table) 
# -----------------------------------------------------

st.header("Threshold sweep (precision/recall/f1)")

if model is not None and X_test_vec is not None and y_test is not None:
    
    try:
        # 創建一個閾值範圍
        thresholds_to_test = np.arange(0.1, 0.91, 0.05) 

        sweep_results = []
        
        # 獲取測試集上的預測機率
        y_proba = model.predict_proba(X_test_vec)[:, 1]
        
        for t in thresholds_to_test:
            t = round(t, 2)
            
            y_pred = (y_proba >= t).astype(int)
            
            p = precision_score(y_test, y_pred, zero_division=0)
            r = recall_score(y_test, y_pred, zero_division=0)
            f = f1_score(y_test, y_pred, zero_division=0)
            
            sweep_results.append({
                'threshold': t,
                'precision': round(p, 4),
                'recall': round(r, 4),
                'f1': round(f, 4)
            })

        sweep_df = pd.DataFrame(sweep_results)
        
        # 顯示表格 (模仿圖 e7d5c3)
        st.dataframe(sweep_df, hide_index=True)
        
    except Exception as e:
        st.error(f"閾值掃描失敗: {e}")
        st.warning("請確保模型和數據已正確載入。")
        
else:
    st.info("閾值掃描表格：請先成功訓練和載入模型。")

st.markdown("---")
# -----------------------------------------------------
# 詞彙分析 (Top Tokens by Class) 
# -----------------------------------------------------

st.header("Token Analysis (詞彙分析)")

if vectorizer_loaded is not None and X_train_vec is not None and y_train is not None:
    
    top_n = st.slider("Top N tokens (熱門 N 詞彙)", 5, 50, 20)
    
    try:
        top_tokens_ham, top_tokens_spam = get_top_tokens(vectorizer_loaded, X_train_vec, y_train, top_n)

        fig_tokens = make_subplots(rows=1, cols=2, 
                                subplot_titles=("Class: ham (非垃圾郵件)", "Class: spam (垃圾郵件)"))

        # Ham 詞彙圖
        fig_tokens.add_trace(go.Bar(
            x=top_tokens_ham['frequency'], 
            y=top_tokens_ham['token'], 
            orientation='h', 
            name='Ham',
            marker_color='#1f77b4'
        ), row=1, col=1)

        # Spam 詞彙圖
        fig_tokens.add_trace(go.Bar(
            x=top_tokens_spam['frequency'], 
            y=top_tokens_spam['token'], 
            orientation='h', 
            name='Spam',
            marker_color='#d62728'
        ), row=1, col=2)

        fig_tokens.update_layout(height=600, showlegend=False, title_text=f"Top {top_n} Tokens by Class (TF-IDF Sum)")
        fig_tokens.update_yaxes(autorange="reversed", row=1, col=1)
        fig_tokens.update_yaxes(autorange="reversed", row=1, col=2)
        fig_tokens.update_xaxes(title_text="frequency (TF-IDF Sum)", row=1, col=1)
        fig_tokens.update_xaxes(title_text="frequency (TF-IDF Sum)", row=1, col=2)
        
        st.plotly_chart(fig_tokens, use_container_width=True)
        
    except Exception as e:
        st.error(f"詞彙分析圖表生成失敗: {e}")
        st.warning("請確保模型和數據已正確載入。")

else:
    st.info("詞彙分析區塊：請先成功訓練和載入模型後，此處才會顯示詞彙分析圖表。")

st.markdown("---")
# -----------------------------------------------------
# 實時推論 (Live Inference) 
# -----------------------------------------------------

st.header("Live Inference (實時推論)")

if model is not None and vectorizer_loaded is not None:
    
    # 按鈕區
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Use spam example"):
            st.session_state['message'] = TEST_SPAM_MSG
    with col_btn2:
        if st.button("Use ham example"):
            st.session_state['message'] = TEST_HAM_MSG

    # 輸入框 (使用 session_state 保持按鈕和輸入框同步)
    if 'message' not in st.session_state:
        st.session_state['message'] = ""

    input_message = st.text_area(
        "Enter a message to classify (輸入要分類的訊息)", 
        st.session_state['message'],
        height=150
    )

    # 預測按鈕
    if st.button("Predict (預測)", key="predict_btn") and input_message:
        
        # 執行預測
        prediction, proba = predict_message(
            input_message, 
            vectorizer_loaded, 
            model, 
            threshold=decision_threshold
        )
        
        st.subheader("Prediction Result (預測結果)")
        
        # 顯示結果
        if prediction == 'spam':
            st.error(f"分類結果：**{prediction.upper()}** (垃圾郵件)")
        else:
            st.success(f"分類結果：**{prediction.upper()}** (非垃圾郵件)")
            
        st.markdown(f"該訊息是 **SPAM** 的機率為: **`{proba:.4f}`**")
        st.markdown(f"使用的決策閾值 (Decision Threshold): **`{decision_threshold:.2f}`**")
else:
    st.info("實時推論區塊：請先成功訓練和載入模型後，此處才會啟用。")