# 變更提案：新增多項式樸素貝葉斯分類器 (Multinomial Naive Bayes)

## 1. 問題陳述 (Problem Statement)
目前的核心機器學習管線（Pipeline）主要集中於邏輯迴歸（Logistic Regression）。鑑於樸素貝葉斯（Naive Bayes）通常是文本分類問題的有力且高效的基準模型，我們需要實作它來建立一個更全面的性能基準比較。

## 2. 提議的變更 (Proposed Change)
將 **多項式樸素貝葉斯分類器 (Multinomial Naive Bayes, MNB)** 實作到模型訓練與評估管線中。

## 3. 實作步驟 (Implementation Steps)
1.  **程式碼修改：** 在 `src/pipeline.py` 中，從 `sklearn.naive_bayes` 匯入 `MultinomialNB` 模組。
2.  **模型訓練：** 新增函數或邏輯來訓練 MNB 模型，使用 TF-IDF 向量化後的文本數據。
3.  **評估：** 使用測試集，計算 MNB 模型的混淆矩陣 (Confusion Matrix)、精確率 (Precision)、召回率 (Recall) 和 F1-score。
4.  **結果報告：** 更新 Streamlit 應用程式 (`app.py`)，使其能夠並排展示 Logistic Regression 和 MNB 模型的性能指標，以供比較。
5.  **模型儲存：** 使用 `pickle` 將訓練好的 MNB 模型儲存到 `models/` 目錄中。

## 4. 驗收標準 (Acceptance Criteria)
* 專案成功訓練並儲存了 MNB 模型。
* MNB 模型的性能指標（特別是 F1-score）被正確計算並記錄。
* Streamlit 演示介面可以清楚地展示 MNB 模型及其評估指標。
* 與 Logistic Regression 模型相比，能夠得出關於 MNB 性能的簡要分析。