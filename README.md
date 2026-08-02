# 🛍️ Flipkart Product Review Sentiment & Rating Predictor (NLP)

An End-to-End **Natural Language Processing (NLP)** web application that predicts product ratings (`Rate`: 1 to 5) based on customer reviews and feedback summaries using TF-IDF tokenization and supervised Machine Learning classifiers.

[![Live App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](YOUR_LIVE_APP_LINK_HERE)

---

## 🖼️ Application Preview

![Flipkart Review Rating Predictor](YOUR_IMAGE_PREVIEW_URL_HERE)

---

## 🔗 Live Application

Access the interactive live web interface:  
👉 **[Click Here to Launch Live Application](YOUR_LIVE_APP_LINK_HERE)**

---

## ⚙️ Data Preprocessing & NLP Pipeline

The input dataset (`flipkart_product.csv`) combines `Review` and `Summary` into a single target feature string `Text`. Custom text normalization is applied prior to tokenization:

1. **Punctuation Removal:** Cleans standard punctuation marks using `str.maketrans`.
2. **Lowercasing:** Standardizes characters to lowercase.
3. **Digit Removal:** Strips numeric digits from input text.
4. **Emoji & Non-ASCII Removal:** Filters non-ASCII characters to retain clean English tokens.
5. **Vectorization:** Transforms normalized text into numeric matrices using serialized `tokenizer.pkl`.

---

## 📁 Repository Structure

```text
├── flipkart_product.csv   # Primary Product Reviews Dataset
├── notebook.ipynb         # EDA, Text Normalization & Model Training
├── app.py                 # Interactive Streamlit Web Application
├── model.pkl              # Serialized Trained Machine Learning Classifier
├── tokenizer.pkl          # Fitted TF-IDF / Vectorizer Object
├── columns.pkl            # Feature Column Schema Definitions
└── README.md              # Documentation

End-to-End Flipkart product review rating predictor powered by NLP & Extra Trees Classifier (83.2% test accuracy). Features custom text preprocessing, TensorFlow Tokenizer sequence padding, and an interactive Streamlit UI designed to replace unstable ANN training architectures.
```
