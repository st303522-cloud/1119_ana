import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 設定網頁標題與版面
st.set_page_config(page_title="中文情緒分析評等系統", layout="centered")
st.title("💬 中文商品評論情緒分析系統")
st.write("請輸入一段產品使用的評論，系統將會預測其 1 至 5 等級的情緒分數！")

# 使用 st.cache_resource 避免重複載入模型以加快速度
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

try:
    with st.spinner("正在載入 AI 模型中，請稍候..."):
        tokenizer, model = load_model()
    st.success("模型載入成功！")
except Exception as e:
    st.error(f"模型載入失敗: {e}")

def analyze_sentiment_5_levels(text):
    if not text or not text.strip():
        return 3

    # 1. 深度學習模型基礎預估
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    base_rating = torch.argmax(probs, dim=-1).item() + 1

    # 2. 中文特徵詞精準加權校正
    very_bad_words = ["垃圾", "爛", "差勁", "別買", "後悔", "失望", "糟糕", "極差", "死當", "欺騙"]
    bad_words = ["不怎麼", "不太", "塑膠感", "陽春", "缺點", "慢", "普通而已", "不好", "不優"]
    neutral_words = ["普通", "還可以", "普普通通", "一般般", "勉強", "尚可", "可接受", "沒感覺", "不期不待", "普通", "便宜"]
    good_words = ["不錯", "好用", "挺好", "滿意", "推薦", "速度快", "舒服", "不賴", "合適"]
    very_good_words = ["太棒了", "完美", "超讚", "極推", "最棒", "無可挑剔", "非常棒", "一流"]

    is_very_bad = any(w in text for w in very_bad_words)
    is_bad = any(w in text for w in bad_words)
    is_neutral = any(w in text for w in neutral_words)
    is_good = any(w in text for w in good_words)
    is_very_good = any(w in text for w in very_good_words)

    if is_very_bad:
        return 1
    elif is_bad:
        return 2
    elif is_neutral:
        return 4 if (is_good or is_very_good) else 3
    elif is_very_good:
        return 5
    elif is_good:
        return 4

    return base_rating

# 建立介面元件
user_input = st.text_area("請輸入產品評論內容:", placeholder="這款產品真的很超值，推薦給大家！", height=150)

# 按下按鈕後執行預測
if st.button("開始進行情緒預測", type="primary"):
    if user_input.strip():
        rating = analyze_sentiment_5_levels(user_input)
        
        # 依分數顯示不同的星號與說明
        star_display = "⭐" * rating
        
        st.subheader("📌 預測結果")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="情緒分數", value=f"{rating} / 5")
        with col2:
            st.write(f"**評級狀態：** {star_display}")
            if rating >= 4:
                st.success("這是一則正面的產品評論！")
            elif rating == 3:
                st.info("這是一則中性、普普通通的評論。")
            else:
                st.error("這是一則負面的產品評論！")
    else:
        st.warning("請輸入有效的評論內容再進行預測！")
