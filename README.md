# 📰 News Category Classifier - Streamlit Frontend

A Streamlit-based web application that classifies news headlines into predefined categories using a fine-tuned BERT model. The app also includes explainable AI features, word cloud visualizations, and real-time RSS feed classification.

---

## 🚀 Features

### 🧪 Headline Classifier
- Input a news headline and classify it into one of the predefined categories using a fine-tuned BERT model.
- Shows prediction confidence as a colorful bar chart.
  
### 🧠 SHAP Explanation (Simulated)
- Provides a simulated SHAP-like explanation showing how each word influenced the model’s decision.
  
### 📊 Word Cloud
- Generates a word cloud from the input headline to visualize key terms.

### 🌍 Live RSS Feed Classification
- Select from trusted sources (BBC, CNN, Reuters, etc.) or add your own RSS URL.
- Automatically classifies headlines from the news feed.

### ☁️ Category Word Clouds
- Groups live RSS headlines into categories and generates a word cloud for each.

---

## 🖼️ Frontend Stack

| Component       | Description                                  |
|----------------|----------------------------------------------|
| **Streamlit**   | Web UI framework used to build the app       |
| **Plotly**      | Interactive charts for visualizing predictions |
| **Matplotlib**  | Used to plot SHAP explanations               |
| **WordCloud**   | To generate word cloud images                |
| **Feedparser**  | Parse and display live news from RSS feeds   |
| **HuggingFace** | Runs classification pipeline in real-time    |

---
## 📸 Screenshots

### 🧪 Classifier Tab
![Classifier Tab](screenshots/News.png)

### 🧠 SHAP Explanation
![SHAP Explanation](screenshots/shap_explanation.png)

### 🌍 RSS Feed Results
![RSS Results](screenshots/Live News feed.png)

### ☁️ Word Cloud
![Word Cloud](screenshots/wordcloud.png)


