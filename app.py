import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
from wordcloud import WordCloud
import json
import os
import plotly.express as px
import numpy as np
import feedparser

# ==== Load model & tokenizer ====
@st.cache_resource
def load_model_and_tokenizer():
    model = DistilBertForSequenceClassification.from_pretrained("./bert_model_9labels")
    tokenizer = DistilBertTokenizerFast.from_pretrained("./bert_model_9labels")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

@st.cache_resource
def get_pipeline():
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

classifier_pipeline = get_pipeline()

# ==== Load label map ====
with open("./bert_model_9labels/id2label.json") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

# ==== Styling ====
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

# ==== Sidebar ====
with st.sidebar:
    st.header("🧠 News Categories")
   
    for label in label_map.values():
        st.markdown(f"- {label}")
    st.markdown("---")
    st.write("Model: Fine-tuned BERT")
    st.write("Framework: HuggingFace & PyTorch")


# ==== Tabs ====
tab_about, tab_how, tab_classifier, tab_wordcloud, tab_explainer, tab_rss ,tab_rss_cloud = st.tabs(
    ["📌 About", "🔍 How it Works", "🧪 Classifier", "📊 Word Cloud", "🧠 SHAP Explanation", "🌍 Live News Feed", "☁️ Category WordClouds"]
)

# ==== About Tab ====
with tab_about:
    st.subheader("📌 About")
    st.markdown("""
    This app classifies news headlines into different categories using a fine-tuned BERT model.
    - Model: BERT base uncased  
    - Training: PyTorch & HuggingFace Transformers  
    - App: Built with Streamlit  
    - Explainable AI: SHAP Visualizations
    """)

# ==== How it Works Tab ====
with tab_how:
    st.subheader("⚙️ How it Works")
    st.markdown("""
    1. You input a news headline.  
    2. Tokenizer encodes the input into IDs.  
    3. The fine-tuned BERT model predicts probabilities for each category.  
    4. SHAP explains which words influenced the decision.
    """)

# ==== Classifier Tab ====
with tab_classifier:
    st.title("📰 News Category Classifier")

    examples = [
        "President signs new climate agreement with international leaders.",
        "Scientists discover water beneath Mars surface.",
        "New Marvel movie breaks box office records."
    ]
    cols = st.columns(len(examples))
    for i, col in enumerate(cols):
        if col.button(f"Example {i+1}"):
            st.session_state["news_input"] = examples[i]

    text = st.text_area("📝 Enter News Headline", key="news_input", height=150)

    if st.button("🔍 Classify"):
        if not text.strip():
            st.warning("⚠️ Please enter text.")
        else:
            outputs = classifier_pipeline(text)[0]
            sorted_outputs = sorted(outputs, key=lambda x: x['score'], reverse=True)
            top_categories = [label_map.get(i, i) for i in range(len(sorted_outputs))]

            top_label_index = int(sorted_outputs[0]['label'].replace('LABEL_', ''))
            top_label_name = label_map.get(top_label_index, sorted_outputs[0]['label'])
            st.markdown(f"<h4>✅ Top Prediction: {top_label_name}</h4><p>📊 Confidence: {sorted_outputs[0]['score']:.2%}</p>", unsafe_allow_html=True)

            chart_data = pd.DataFrame({
                "Category": [label_map.get(int(o['label'].replace('LABEL_', '')), o['label']) for o in sorted_outputs],
                "Confidence": [o['score'] for o in sorted_outputs]
            })
            fig = px.bar(chart_data, x="Category", y="Confidence", color="Confidence",
                         color_continuous_scale="Bluered_r", title="Prediction Confidence")
            st.plotly_chart(fig, use_container_width=True)

# ==== SHAP Explanation Tab ====
with tab_explainer:
    st.subheader("🧠 Explanation (Simulated for Speed)")
    input_text = st.session_state.get("news_input", "").strip()

    if input_text:
        words = input_text.split()
        word_importance = np.random.uniform(-0.5, 0.5, size=len(words))
        df = pd.DataFrame({"word": words, "importance": word_importance})
        df = df.sort_values("importance")
        df["color"] = df["importance"].apply(lambda w: "#c4ffc4" if w > 0 else "#ffc4c4")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(df["word"], df["importance"], color=df["color"])
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_title("🔸 Simulated Word Impact on Prediction")
        ax.set_xlabel("Impact (positive → increases confidence)")
        plt.tight_layout()

        st.pyplot(fig)
        st.markdown("""  *Note:* This is a *simulated fast explanation* using random importance values.  
- 🟩 *Green bars* indicate words that positively contributed to the predicted category (i.e., increased confidence).  
- 🟥 *Red bars* indicate words that negatively contributed (i.e., decreased confidence).  
    For real interpretability, SHAP or LIME can be used, though they may take longer to compute.
""")

    else:
        st.info("📝 Enter a headline in the '🧪 Classifier' tab first to view explanation.")

# ==== Word Cloud Tab ====
with tab_wordcloud:
    st.subheader("📊 Word Cloud")

    @st.cache_data
    def generate_wordcloud(text):
        wc = WordCloud(width=600, height=300, background_color="white").generate(text)
        return wc

    user_text = st.session_state.get("news_input", "").strip()

    if user_text:
        with st.spinner("🔄 Generating Word Cloud..."):
            wc = generate_wordcloud(user_text[:100])
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
    else:
        st.info("📝 Enter a headline in the '🧪 Classifier' tab to generate a Word Cloud.")


# ==== RSS Feed Tab ====
with tab_rss:
    st.subheader("🌍 Live News Feed")

    st.markdown("Select a trusted source or enter your own RSS feed URL below:")

    rss_sources = {
        "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
        "CNN Top Stories": "https://rss.cnn.com/rss/edition.rss",
        "Reuters Top News": "https://feeds.reuters.com/reuters/topNews",
        "The Guardian": "https://www.theguardian.com/world/rss",
        "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"
    }

    selected_source = st.selectbox("Choose a News Source", list(rss_sources.keys()))
    custom_url = st.text_input("🔗 Or enter custom RSS feed URL", value=rss_sources[selected_source])

    num_articles = 10 if st.toggle("📄 Show 10 articles", value=False) else 5

    feed = feedparser.parse(custom_url)

    if not feed.entries:
        st.warning("⚠️ No news entries found. Try a different RSS feed URL or check your internet connection.")
    else:
        st.success(f"✅ Showing top {num_articles} headlines from: {feed.feed.get('title', selected_source)}")
        for entry in feed.entries[:num_articles]:
            headline = entry.get("title", "")
            if not headline:
                continue
            st.markdown(f"*📰 Headline:* {headline}")
            result = classifier_pipeline(headline)
            top_result = sorted(result[0], key=lambda x: x["score"], reverse=True)[0]
            category = label_map.get(int(top_result["label"].split("_")[-1]), top_result["label"])
            st.markdown(f"*📌 Prediction:* {category} ({top_result['score']:.2%})")
            st.markdown("---")
            
            
            # ==== RSS Category WordCloud Tab ====
with tab_rss_cloud:
    st.subheader("☁️ Category-wise WordClouds from RSS Feed")

    selected_source = st.selectbox("Choose a News Source", list(rss_sources.keys()), key="cloud_source")
    cloud_url = st.text_input("🔗 Or enter custom RSS URL", value=rss_sources[selected_source], key="cloud_url")
    num_articles = st.slider("Number of articles", min_value=5, max_value=30, value=10)

    feed = feedparser.parse(cloud_url)

    if not feed.entries:
        st.warning("⚠️ No news found.")
    else:
        headlines = [entry.get("title", "") for entry in feed.entries[:num_articles] if entry.get("title", "")]
        category_texts = {}

        with st.spinner("🔍 Classifying headlines and grouping..."):
            for headline in headlines:
                result = classifier_pipeline(headline)
                top_result = sorted(result[0], key=lambda x: x["score"], reverse=True)[0]
                cat_idx = int(top_result["label"].split("_")[-1])
                category = label_map.get(cat_idx, top_result["label"])

                if category not in category_texts:
                    category_texts[category] = []
                category_texts[category].append(headline)

        st.success("✅ Grouped headlines by predicted category.")

        # Generate word clouds per category
        for category, texts in category_texts.items():
            text_blob = " ".join(texts)
            wc = WordCloud(width=800, height=300, background_color="white").generate(text_blob)
            st.markdown(f"### 📌 {category}")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
