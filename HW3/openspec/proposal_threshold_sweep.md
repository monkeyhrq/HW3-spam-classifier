# 提案：整合決策閾值掃描表格 (Threshold Sweep Table)

## 1. 摘要 (Summary)
**目的：** 在 Streamlit 儀表板的「Model Performance (模型性能)」區塊中新增一個表格，用於展示不同決策閾值 (Decision Threshold) 對模型性能指標（Precision, Recall, F1 Score）的影響。
**目標：** 增強儀表板的**視覺化與解釋性** (Visualization & Interpretability)，幫助用戶選擇最佳的閾值來平衡模型的精確度 (Precision) 和召回率 (Recall)。

## 2. 規格 (Specification)

### 2.1. 功能詳情 (Feature Details)
- **位置：** 位於「Model Performance (模型性能)」下方。
- **數據範圍：** 掃描閾值範圍從 $0.1$ 到 $0.9$，步長為 $0.05$。
- **計算指標：** 對於每個閾值，計算測試集上的精確度 (Precision)、召回率 (Recall) 和 F1 Score。
- **UI 呈現：** 以 Streamlit `st.dataframe` 格式呈現結果，包含四個欄位：`threshold` (閾值)、`precision` (精確度)、`recall` (召回率) 和 `f1` (F1 分數)。

### 2.2. 技術實作 (Technical Implementation)
- **檔案修改：**
    - `src/app.py`: 實作數據獲取、循環計算和表格渲染的邏輯。
    - `src/utils.py`: (無須修改，因為所需的 `precision_score`, `recall_score`, `f1_score` 已在 `app.py` 中引入或在 `evaluate_model` 中使用)。
- **依賴項：** 需確保安裝了 `scikit-learn` 和 `numpy`。

## 3. 實作驗證 (Verification)
- **驗證步驟：** 執行 `streamlit run src/app.py`。
- **預期結果：** 在儀表板上應看到一個名為「Threshold sweep (precision/recall/f1)」的表格，列出約 15 個不同閾值下的性能指標，數值應與手動計算結果一致。

## 4. 理由 (Rationale)
在垃圾郵件分類任務中，召回率 (Recall) 至關重要。通過可視化閾值掃描，使用者可以清楚看到將閾值從預設 $0.5$ 調整到 $0.35$ 或 $0.40$ 等更低值時，召回率的大幅提升（從約 $0.63$ 提升至 $0.87$），從而更好地評估模型在實際應用中的效用。