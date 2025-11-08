# 專案概覽 (Project Overview)

## 1. 專案名稱 (Project Name)
Email Spam Classification Dashboard

## 2. 專案目標 (Project Goal)
本專案的目標是基於簡訊數據集 (SMS Dataset) 建立一個垃圾郵件（Spam/Ham）分類模型，並使用 Streamlit 框架開發一個互動式儀表板。該儀表板需展示模型的性能指標、數據分佈、詞彙分析，並提供實時推論和決策閾值調整功能，以符合 OpenSpec (Spec-Driven Development) 的要求。

## 3. 技術棧 (Tech Stack)

| 類別 (Category) | 技術 (Technology) | 用途 (Purpose) |
| :--- | :--- | :--- |
| **程式語言** | Python 3.8+ | 主要開發語言 |
| **數據處理** | Pandas, NumPy | 數據加載、清理和數值運算 |
| **機器學習** | Scikit-learn | 模型訓練 (Logistic Regression), 特徵工程 (TF-IDF), 性能評估 (Metrics) |
| **視覺化** | Streamlit, Plotly | 互動式儀表板 (UI) 構建和高性能圖表繪製 |
| **檔案管理** | Joblib, pathlib | 模型和 Vectorizer 的持久化保存與跨平台路徑管理 |
| **工作流程** | OpenSpec, AI Coding CLI | 規範驅動的開發和 AI 輔助開發追蹤 |

## 4. 專案慣例 (Conventions)

### 4.1. 目錄結構 (Directory Structure)
專案遵循標準的目錄結構，確保數據、原始碼和模型分離。
- **`src/`**: 包含應用程式邏輯 (`app.py`, `utils.py`)。
- **`Chapter03/datasets/`**: 存放原始數據集 (`sms_spam_no_header.csv`)。
- **`models/`**: 存放訓練好的模型和 Vectorizer。
- **`openspec/`**: 存放 OpenSpec 相關文件 (`project.md`, `proposal`, `AGENTS.md`)。

### 4.2. 編碼與命名 (Encoding & Naming)
- **程式碼風格**: 遵循 Python PEP 8 規範。
- **數據編碼**: 數據集使用 `latin-1` 或 `UTF-8` 進行讀取，以確保兼容性。
- **模型儲存**: 使用 `joblib` 序列化模型，副檔名為 `.joblib`。

### 4.3. 數據前處理 (Data Preprocessing)
- **特徵工程**: 使用 **TF-IDF Vectorizer** 將清理後的文本轉換為數值特徵。
- **文本清理**: 包含轉換為小寫、移除標點符號和數字。

---

接下來，我們需要創建 **功能變更提案 (Change Proposal)**。您想將哪一個主要的儀表板功能（例如：**閾值掃描表格**、**詞彙分析圖** 或 **實時推論**）作為您的提案主題呢？