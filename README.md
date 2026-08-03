# 🛍️ Flipkart Product Review Sentiment & Rating Predictor (NLP)

An End-to-End **Natural Language Processing (NLP)** web application that predicts product ratings (`Rate`: 1 to 5) based on customer reviews and feedback summaries using TF-IDF tokenization and supervised Machine Learning classifiers.

[![Live App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](YOUR_LIVE_APP_LINK_HERE)

---

## 🖼️ Application Preview

![Flipkart Review Rating Predictor](YOUR_IMAGE_PREVIEW_URL_HERE)

---

## 🔗 Live Application

Access the interactive live web application:  
👉 **[Click Here to Launch Live Application](YOUR_LIVE_APP_LINK_HERE)**

---

## 💡 Model Selection & Engineering Challenges

During the development and experimental phase, multiple model architectures were evaluated:

### ❌ Artificial Neural Network (ANN) Limitations

- **High Loss & Accuracy Fluctuations:** Training an ANN architecture led to severe epoch-to-epoch instability in both accuracy and training loss.
- **Validation Loss Stagnation:** The test/validation loss failed to converge effectively, indicating that deep neural layers were struggling to generalize on the sparse TF-IDF text matrix without overfitting.
- **Decision:** The ANN approach was discarded due to unpredictable performance and poor generalization on unseen test data.

### 🏆 Machine Learning Approach (Selected Solution)

- **Stable Generalization:** Transitioned to traditional Machine Learning classification algorithms which provided consistent convergence.
- **High Test Performance:** The final trained ML model achieved an **83% Test Accuracy**, successfully resolving the variance/loss issues observed in the ANN model while ensuring fast, reliable inference for real-time web deployment.

---

## ⚙️ Data Preprocessing & NLP Pipeline

The input dataset (`flipkart_product.csv`) combines `Review` and `Summary` into a single feature string `Text`. Custom text normalization is applied prior to tokenization:

1. **Punctuation Removal:** Cleans standard punctuation marks using `str.maketrans`.
2. **Lowercasing:** Standardizes characters to lowercase.
3. **Digit Removal:** Strips numeric digits from input text.
4. **Emoji & Non-ASCII Removal:** Filters non-ASCII characters to retain clean English tokens.
5. **Vectorization:** Transforms normalized text into numeric matrices using the serialized `tokenizer.pkl`.

---

## 📁 Repository Structure

```text
├── flipkart_product.csv   # Primary Product Reviews Dataset
├── .gitignore             # Git Ignore File
├── .gitattributes         # Git Attributes File
├── notebook.ipynb         # EDA, Text Normalization & Model Experiments
├── app.py                 # Interactive Streamlit Web Application
├── model.pkl              # Serialized Trained Machine Learning Classifier (83% Test Acc)
├── tokenizer.pkl          # Fitted TF-IDF / Vectorizer Object
├── columns.pkl            # Feature Column Schema Definitions
└── README.md              # Documentation
```

---

```bash
git clone https://github.com/amirsohail100/Flipkart-Product-Review-Sentiment-Rating-Predictor-NLP-.git
```

```bash
cd Flipkart-Product-Review-Sentiment-Rating-Predictor-NLP-
```

---

```bash
streamlit run app.py
```

---

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the MIT License.

## 📝 Author

## 👤 **Amir Sohail**

---

End-to-End Flipkart product review rating predictor powered by NLP & Extra Trees Classifier (83.2% test accuracy). Features custom text preprocessing, TensorFlow Tokenizer sequence padding, and an interactive Streamlit UI designed to replace unstable ANN training architectures.
