# 📧 HW3：Email Spam Classification Project
**Spec-Driven Development (OpenSpec) 專案實作與 Streamlit 視覺化儀表板**

本專案實作了一個基於 TF-IDF 特徵和 Logistic Regression 模型的簡訊垃圾郵件（Spam/Ham）分類器。整個專案按照 OpenSpec (Spec-Driven Development) 工作流程進行，並使用 Streamlit 建立互動式儀表板進行模型性能與數據的視覺化和解釋性分析。

---

## 🚀 1. 專案設定與運行 (Setup & Run)

### 1.1 環境依賴項 (`requirements.txt`)

請先確保您的環境中安裝了 Python 3.8+。所有必要的依賴套件已列於 `requirements.txt` 中。

**請創建以下 `requirements.txt` 檔案：**

```txt
# requirements.txt
streamlit>=1.0.0
pandas>=1.0.0
numpy>=1.20.0
scikit-learn>=1.0.0
plotly>=5.0.0
joblib>=1.0.0