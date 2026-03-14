import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
import random
import google.generativeai as genai

# 頁面設定
st.set_page_config(page_title="工廠異常檢測系統", layout="wide")

# 載入模型與資料
@st.cache_resource
def load_model():
    return joblib.load('iso_forest.joblib')

@st.cache_data
def load_data():
    return pd.read_csv('train_FD001_clean.csv')

try:
    model = load_model()
    df = load_data()
except FileNotFoundError:
    st.error("找不到檔案，請確認 iso_forest.joblib 和 train_FD001_clean.csv 在同目錄下。")
    st.stop()

# 健康度換算模組 
def calculate_health_index(score, threshold):
    """將 Risk Score 轉換成 0~100 的健康度"""
    if score <= 0:
        return 100
    elif score < threshold:
        # 在安全範圍內，健康度從 99 降到 60
        return int(100 - (score / threshold) * 40)
    else:
        # 超過 threshold (異常)，健康度從 59 快速下降至 0
        return max(0, int(59 - ((score - threshold) / threshold) * 50))

# 側邊欄：導覽列
st.sidebar.title("異常檢測系統")
st.sidebar.subheader("Smart Fleet Management")

# 使用者輸入 API Key
api_key = st.sidebar.text_input("輸入 Gemini API Key (解鎖 GenAI 報告)", type="password")
if api_key:
    genai.configure(api_key=api_key)

# 使用者選擇模式
mode = st.sidebar.radio("選擇檢視模式 (View Mode)", ["全廠設備總覽 (Fleet Dashboard)", "單機詳細監控 (Single Device Detail)"])

# 防止按鈕異常
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = mode

# 現在的模式 <> 上一刻紀錄的模式
if st.session_state.last_mode != mode:
    st.session_state.last_mode = mode # 更新紀錄
    st.rerun() # 強制刷新

# 共用的閾值設定 (讓全廠標準一致)
threshold = st.sidebar.slider("異常判定閥值 (Threshold)", min_value=0.0, max_value=0.1, value=0.025, step=0.001)

# 模式：全廠設備總覽 
if mode == "全廠設備總覽 (Fleet Dashboard)":
    st.title("全廠設備健康狀態總覽")
    
    # 按下這顆按鈕才開始跑迴圈。
    start_scan = st.button("開始全廠掃描", type="secondary")

    if start_scan:
        # --- 步驟 1: 快速掃描全廠 100 台設備的「最後狀態」 ---
        last_status_list = []
        
        # 找出所有的 Engine ID
        engine_ids = df['id'].unique()
        features = [col for col in df.columns if col.startswith('s')]
        
        # 建立進度條
        progress_text = "正在掃描全廠設備狀態..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, eid in enumerate(engine_ids):
            # 撈出該設備的資料
            engine_data = df[df['id'] == eid]
            
            # 模擬「隨機時間點」
            max_cycle = engine_data['cycle'].max()
            random_cycle = random.randint(int(max_cycle * 0.5), max_cycle)
            
            # 抓出那個「隨機時間點」的數據
            current_record = engine_data[engine_data['cycle'] == random_cycle]
            features_data = current_record[features]
            
            # 丟進模型算分數
            score = -model.decision_function(features_data)[0]
            
            # 判斷狀態
            status = "Critical 🚨" if score > threshold else "Healthy ✅"
            
            last_status_list.append({
                "Device ID": eid,
                "Cycle": random_cycle,
                "Risk Score": round(score, 4),
                "Status": status
            })
                        
            # 更新進度條
            my_bar.progress((i + 1) / len(engine_ids), text=progress_text)
            
        my_bar.empty() # 跑完把進度條藏起來
        
        # --- 步驟 2: 顯示結果 ---
        status_df = pd.DataFrame(last_status_list)
        
        # 顯示關鍵指標 (KPI)
        total_machines = len(engine_ids)
        critical_machines = status_df[status_df['Status'] == "Critical 🚨"].shape[0]
        healthy_machines = total_machines - critical_machines
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("監控設備總數", f"{total_machines} 台")
        kpi2.metric("✅ 正常運轉設備", f"{healthy_machines} 台")
        kpi3.metric("🚨 異常告警設備", f"{critical_machines} 台", delta="-需立即檢查", delta_color="inverse")
        
        st.divider()
        
        # 列出風險最高的設備
        st.subheader("⚠️ 高風險設備清單 (Top Critical Machines)")
        critical_df = status_df[status_df['Status'] == "Critical 🚨"].sort_values(by="Risk Score", ascending=False)
        
        if not critical_df.empty:
            st.dataframe(critical_df.style.applymap(lambda x: 'background-color: #ff4b4b; color: white' if x == 'Critical 🚨' else '', subset=['Status']), use_container_width=True)
        else:
            st.success("目前全廠無異常設備！")
            
        with st.expander("查看全廠完整清單"):
            st.dataframe(status_df)
            
    else:
        # 當還沒按下按鈕時的提示畫面
        st.info("請按下上方按鈕，啟動 AI 全廠設備診斷。")
# ---------------------------------------------------------
# 5. 模式 B：單機詳細監控 (Single Device Detail) - 原本的功能
# ---------------------------------------------------------
elif mode == "單機詳細監控 (Single Device Detail)":
    engine_ids = df['id'].unique()
    selected_engine = st.sidebar.selectbox("選擇要檢查的設備 (Select ID)", engine_ids)
    
    st.title(f"設備 #{selected_engine} 詳細診斷")
    
    # 以下就是原本的 v2.0 程式碼
    speed = st.sidebar.slider("模擬速度 (秒/Cycle)", 0.01, 0.5, 0.05)
    
    engine_data = df[df['id'] == selected_engine]
    features = [col for col in df.columns if col.startswith('s')]
    X_test = engine_data[features]
    scores = -model.decision_function(X_test)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        chart_ph = st.empty()
    with col2:
        alert_ph = st.empty()
        metric_ph = st.empty()
        
    start_btn = st.button("啟動即時診斷模擬")
    
    # 初始化 Session State 用來存「診斷歷史」
    if 'diagnosis_history' not in st.session_state:
        st.session_state.diagnosis_history = []
    
    # 建立一個變數來控制 Review 模式的索引
    if 'review_index' not in st.session_state:
        st.session_state.review_index = 0

    if start_btn:
        # 重置歷史紀錄
        st.session_state.diagnosis_history = []
        
        # 預先計算全廠的健康基準線
        baseline_mean = df[features].mean()
        baseline_std = df[features].std()

        history_cycle = []
        history_score = []
        
        # 建立畫面容器
        diagnosis_container = st.empty() # 這裡改成 empty，確保每次都只顯示一張

        # --- 階段一：即時模擬迴圈 ---
        for i in range(len(engine_data)):
            current_cycle = engine_data.iloc[i]['cycle']
            current_record = engine_data.iloc[[i]][features]
            current_score = -model.decision_function(current_record)[0]
            
            history_cycle.append(current_cycle)
            history_score.append(current_score)
            
            # 1. 畫趨勢圖
            fig, ax = plt.subplots(figsize=(10, 3.5))
            if len(history_cycle) > 50:
                ax.plot(history_cycle[-50:], history_score[-50:], color='#ff4b4b', linewidth=2)
            else:
                ax.plot(history_cycle, history_score, color='#ff4b4b', linewidth=2)
            ax.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
            ax.set_title(f"Real-time Anomaly Score (Engine #{selected_engine})")
            ax.set_ylim(min(scores)-0.05, max(scores)+0.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            chart_ph.pyplot(fig)
            plt.close(fig)
            
            # 2. 更新數據 (加入健康度指數)
            health_pct = calculate_health_index(current_score, threshold)
            
            # 決定燈號顏色
            if health_pct >= 80:
                h_color = "🟢 良好"
            elif health_pct >= 60:
                h_color = "🟡 警告"
            else:
                h_color = "🔴 危險"
                
            with metric_ph.container():
                st.metric("Cycle", int(current_cycle))
                # 顯示 100% 格式，並把原始 Score 放在下方作為參考
                st.metric("設備健康度 (Health)", f"{health_pct}% ({h_color})", 
                          delta=f"Risk Score: {current_score:.4f}", 
                          delta_color="inverse" if current_score > threshold else "normal")
            
            # 3. 異常偵測與資料儲存
            if current_score > threshold:
                alert_ph.error(f"異常警告！(Score: {current_score:.3f})")
                
                # 計算歸因
                delta = (current_record.iloc[0] - baseline_mean) / baseline_std
                top_3_causes = delta.abs().sort_values(ascending=False).head(3)
                top_sensor = top_3_causes.index[0]
                
                # [關鍵] 把這次的診斷結果「存起來」，而不是只印出來
                diagnosis_record = {
                    "cycle": int(current_cycle),
                    "score": current_score,
                    "top_3": top_3_causes, # 這是 Series
                    "top_sensor": top_sensor
                }
                st.session_state.diagnosis_history.append(diagnosis_record)
                
                # --- 即時顯示當下的診斷 (Live View) ---
                # 這裡只顯示「最新」的一筆，不會疊加
                with diagnosis_container.container():
                    st.markdown(f"### AI 智慧診斷報告 -- Cycle {int(current_cycle)}")
                    dc1, dc2 = st.columns([1, 1])
                    with dc1:
                        st.bar_chart(top_3_causes, color="#ff4b4b")
                    with dc2:
                        st.info(f"⚠️ 檢測到 **{top_sensor}** 異常！\n\n(模擬結束後可檢視詳細建議)")

            else:
                alert_ph.success("✅ 運轉正常 (System Healthy)")
                diagnosis_container.empty() # 正常時清空診斷區
                
            time.sleep(speed)
        
        st.success("✅ 模擬結束！現在您可以透過下方控制面板回顧所有異常報告。")

    # --- 階段二：模擬結束後的「時光機回顧模式」 ---
    # 這個區塊在模擬結束後會一直存在，讓使用者操作
    
    if len(st.session_state.diagnosis_history) > 0:
        st.divider()
        st.markdown("## ⏪ 異常診斷回溯 (Historical Review)")
        
        # 1. 建立控制列 (上一頁 | 下拉選單 | 下一頁)
        c_prev, c_sel, c_next = st.columns([1, 2, 1])
        
        # 取得總共有幾筆異常
        total_records = len(st.session_state.diagnosis_history)
        
        # [控制邏輯] 下一頁按鈕
        with c_next:
            if st.button("下一筆 ▶️", key="btn_next"):
                if st.session_state.review_index < total_records - 1:
                    st.session_state.review_index += 1

        # [控制邏輯] 上一頁按鈕
        with c_prev:
            if st.button("◀️ 上一筆", key="btn_prev"):
                if st.session_state.review_index > 0:
                    st.session_state.review_index -= 1
        
        # [控制邏輯] 下拉選單 (連動 review_index)
        # 製作選單選項: "Cycle 35 (Score: 0.12)", "Cycle 36..."
        options = [f"Cycle {rec['cycle']}" for rec in st.session_state.diagnosis_history]
        
        with c_sel:
            # 1. 移除原本的 key 參數，避免 Streamlit 內部狀態打架
            selected_option = st.selectbox(
                "選擇要檢視的異常週期", 
                options, 
                index=st.session_state.review_index
            )
            
            # 2. 取得下拉選單目前選到的項目的 index
            selected_idx = options.index(selected_option)
            
            # 3. 只有當「選單選到的值」跟「現在系統紀錄的 index」不一樣時，才去更新
            if st.session_state.review_index != selected_idx:
                st.session_state.review_index = selected_idx
                st.rerun() # 強制刷新畫面，讓按鈕和選單完美同步

        # 2. 顯示選定週期的詳細報告 (這就是你原本的診斷介面)
        record = st.session_state.diagnosis_history[st.session_state.review_index]
        
        st.markdown(f"### Cycle {record['cycle']} 詳細診斷報告")
        
        rc1, rc2 = st.columns([1, 1])
        
        with rc1:
            st.markdown("**異常歸因分析 (Top Causes)**")
            st.bar_chart(record['top_3'], color="#ff4b4b")
            
        with rc2:
            st.markdown("**維修建議 (Prescriptive Action)**")
            top_sensor = record['top_sensor']
            
            # 把原本的一長串 if-else 建議邏輯搬來這裡
            if top_sensor in ['s2', 's3', 's4', 's11', 's17']:
                st.warning(f"⚠️ **檢測到 {top_sensor} (溫度/壓力) 異常升高！**")
                st.info("💡 建議措施：\n1. 檢查冷卻系統循環是否堵塞。\n2. 檢查燃油噴嘴是否霧化異常。\n3. 安排熱顯像儀檢測。")
            elif top_sensor in ['s7', 's12', 's20', 's21']:
                st.warning(f"⚠️ **檢測到 {top_sensor} (液壓/冷卻流) 異常波動！**")
                st.info("💡 建議措施：\n1. 檢查管路是否有洩漏。\n2. 確認泵浦加壓效率。\n3. 檢查過濾器是否需更換。")
            elif top_sensor in ['s8', 's9', 's13', 's14']:
                st.warning(f"⚠️ **檢測到 {top_sensor} (轉速/振動) 不穩定！**")
                st.info("💡 建議措施：\n1. 檢查軸承 (Bearing) 是否磨損。\n2. 執行振動頻譜分析。\n3. 確認動平衡狀態。")
            else:
                st.warning(f"⚠️ **檢測到 {top_sensor} 數值異常！**")
                st.info("💡 建議措施：請下載詳細 Log 並聯繫設備原廠進行二級診斷。")

            #  GenAI 專家深度分析模組
            st.divider()
            st.markdown("### ✨ GenAI 專家深度分析報告")
            
            if api_key:
                # 為了避免重複點擊，加上 key
                if st.button("產出 AI 專家客製化診斷報告", key=f"genai_btn_{record['cycle']}"):
                    with st.spinner("正在分析數據並撰寫報告中..."):
                        try:
                            # 準備給 LLM 的 Prompt (提示詞)
                            # 這裡把前三名異常感測器的名稱串起來
                            sensor_list = ", ".join(record['top_3'].index.tolist()) 
                            
                            prompt = f"""
                            你是一位擁有 20 年經驗的科技廠務資深設備工程師。
                            目前戰情室的預知保養系統 (Isolation Forest 模型) 偵測到「第 {selected_engine} 號設備」在「第 {record['cycle']} 週期」出現高風險異常。
                            該筆數據的異常風險分數為 {record['score']}。
                            偏離健康基準線最嚴重的前三個感測器為：{sensor_list} (其中首要主因是 {top_sensor})。
                            
                            請根據上述資訊，撰寫一段約 150 字的專業維修指導報告給第一線維修人員。
                            報告需包含以下結構：
                            1. 【數據解讀】：解釋這幾個感測器同時異常代表什麼物理現象。
                            2. 【可能肇因】：推測最可能的根本原因 (Root Cause)。
                            3. 【行動指示】：給出明確、具體、有先後順序的排查與維修步驟。
                            
                            語氣請保持專業、嚴謹、且具備急迫性。
                            """
                            
                            # 呼叫 Gemini 模型
                            model_gen = genai.GenerativeModel('gemini-2.5-flash')
                            response = model_gen.generate_content(prompt)
                            
                            st.success("報告生成完畢！")
                            # 顯示生成的報告
                            st.info(response.text)
                            
                        except Exception as e:
                            st.error(f"API 呼叫失敗，請檢查連線或 API Key 是否正確。錯誤訊息: {e}")
            else:
                st.warning("請先在左側邊欄輸入 Gemini API Key 以解鎖此進階功能。")

        # ---------------------------------------------------------
        # [新增] 一鍵匯出數位檢修工單 (Export Work Order)
        # ---------------------------------------------------------
        st.divider()
        st.markdown("### 🖨️ 數位檢修工單輸出")
        
        # 1. 組合工單的文字內容
        report_text = f"""=========================================
        [AI 預知保養 - 設備異常檢修工單]
        =========================================
        ► 設備編號: Engine #{selected_engine}
        ► 異常發生週期 (Cycle): {record['cycle']}
        ► 風險分數 (Risk Score): {record['score']:.4f}

        【AI 異常歸因分析】
        首要異常感測器: {top_sensor}

        【系統維修建議】
        """
        # 2. 根據剛剛的 top_sensor 自動填寫對應的建議
        if top_sensor in ['s2', 's3', 's4', 's11', 's17']:
            report_text += "狀態：檢測到溫度/壓力異常升高！\n建議措施：\n1. 檢查冷卻系統循環是否堵塞。\n2. 檢查燃油噴嘴是否霧化異常。\n3. 安排熱顯像儀檢測。\n"
        elif top_sensor in ['s7', 's12', 's20', 's21']:
            report_text += "狀態：檢測到液壓/冷卻流異常波動！\n建議措施：\n1. 檢查管路是否有洩漏。\n2. 確認泵浦加壓效率。\n3. 檢查過濾器是否需更換。\n"
        elif top_sensor in ['s8', 's9', 's13', 's14']:
            report_text += "狀態：檢測到轉速/振動不穩定！\n建議措施：\n1. 檢查軸承 (Bearing) 是否磨損。\n2. 執行振動頻譜分析。\n3. 確認動平衡狀態。\n"
        else:
            report_text += "狀態：檢測到數值異常！\n建議措施：請下載詳細 Log 並聯繫設備原廠進行二級診斷。\n"
            
        report_text += "\n\n現場維修人員簽核：____________________  日期：____________________"
        
        # 3. 產生 Streamlit 下載按鈕
        st.download_button(
            label="匯出工單 (Download Work Order .txt)",
            data=report_text,
            file_name=f"WorkOrder_Engine{selected_engine}_Cycle{record['cycle']}.txt",
            mime="text/plain",
            type="primary"
        )
            
        st.info("模擬結束。")