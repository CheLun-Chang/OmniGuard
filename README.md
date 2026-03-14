# OmniGuard: AI 雙引擎機群預知保養與決策系統

專案簡介
本專案旨在解決工業製造中「無預警停機成本高昂」與「AI 模型缺乏解釋性」的痛點。透過結合**機器學習（預測性 AI）**與**大語言模型（生成式 AI）**，打造出一套具備高度商業落地價值的 AIOps 智慧廠務監控系統。

核心功能
1. 全廠健康總覽：一鍵掃描全廠設備狀態，將演算法分數轉化為直覺的 0~100%「設備健康度指數」。
2. 數位孿生與 XAI 歸因：即時推論感測器數據，觸發警報時透過 Z-Score 分析瞬間抓出 Top 3 異常感測器，打破 AI 黑盒子。
3. GenAI 虛擬廠務顧問：串接 Gemini 2.5 Pro API，根據異常特徵動態生成「專家深度分析報告」，提供具體排查步驟。
4. 自動派工系統：內建異常回溯面板，支援一鍵匯出標準化數位檢修工單，打通 Data-to-Action 閉環。

使用技術
前端框架: Streamlit
機器學習: scikit-learn (Isolation Forest)
生成式 AI: Google Gemini API
資料處理: Pandas, Matplotlib

如何在本地端執行
1. 安裝相依套件：`pip install -r requirements.txt`
2. 啟動戰情室：`streamlit run app.py`
3. 於左側邊欄輸入您的 Gemini API Key 即可解鎖 GenAI 專家功能。
