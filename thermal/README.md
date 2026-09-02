# 熱線 (Thermal Line) 分析腳本

本目錄包含電池熱特徵分析（W1d、W2b、W2c）的實作腳本。

## 執行順序與上游依賴

### 1. W2b_process.py
- **功能**：執行雙路徑（放電段 vs 靜置段）對接與參數擬合，產生各循環的熱參數。
- **輸入資料**：NASA 電池數據 (`cleaned_dataset/`)
- **產出檔案**：`w2b_fit_by_cycle.csv` 等多個 CSV 與圖表，預設輸出至 `output/`。

### 2. W2c_process.py
- **功能**：讀取 W2b 擬合結果，執行多起點收斂性、T_amb 掃描與 dU/dT 敏感度分析，並繪製診斷圖表。
- **輸入資料**：NASA 電池數據 (`cleaned_dataset/`)
- **上游依賴**：必須讀取由 `W2b_process.py` 所產生的 `w2b_fit_by_cycle.csv`。
- **執行順序**：必須在 `W2b_process.py` 執行完畢後執行。

### 3. W1d_process.py
- **功能**：評估單集總熱模型的 T_amb 假設與殘差分佈，產生熱特徵的最終報告文件。
- **上游依賴**：需要讀取 `tau_by_cycle_v3.csv`，此檔案是更上游腳本 (`W1c_process.py`) 的產出，目前作為中間產物存放於 `data/` 供本腳本讀取。
- **執行順序：目前無法獨立執行。** 本腳本需要兩個不同來源——上游產出的
`tau_by_cycle_v3.csv`，以及 NASA 的原始逐循環 csv——而現行介面只有一個 `--data-dir`，
指不到兩處。這是已知缺陷，尚未修正。

## NASA 資料集來源與取得說明

本目錄腳本依賴 NASA 電池老化資料集（使用經清洗之 `cleaned_dataset/` 格式）。
根據目錄下的原始 `README.md` 說明，本專案使用的並非原始的 `.mat` 檔案，而是「經由第三方（如 Kaggle 上游）清洗與轉換過的 cleaned_dataset/」。
由於原文件未註明確切的 Kaggle 網址或發布者，實際取得來源目前**待確認**。
（注意：專案中原存的 `B0005.mat` 經查證為失效的 HTML 網頁，無法用於資料重建。）
