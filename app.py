import streamlit as st
import pickle
import string
import pandas as pd
import numpy as np

# Import TensorFlow sequence padder for TF Tokenizer
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from keras.preprocessing.sequence import pad_sequences

# --- Page Configuration ---
st.set_page_config(
    page_title="Flipkart Review Rating Predictor",
    page_icon="🛍️",
    layout="centered"
)

# --- NLP Preprocessing Functions ---
def remove_punc(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def tolower(txt):
    return txt.lower()

def remove_num(txt):
    new = ""
    for i in txt:
        if not i.isdigit():
            new += i
    return new

def remove_emoj(txt):
    new = ""
    for i in txt:
        if i.isascii():
            new += i
    return new

def preprocess_text(txt):
    txt = remove_punc(txt)
    txt = tolower(txt)
    txt = remove_num(txt)
    txt = remove_emoj(txt)
    return txt

# --- Load Models & Assets ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("columns.pkl", "rb") as f:
            columns = pickle.load(f)
        return model, tokenizer, columns, True
    except Exception as e:
        return None, None, None, False

model, tokenizer, columns, assets_loaded = load_assets()

# --- UI Layout ---
st.title("🛍️ Flipkart Product Sentiment & Rating Predictor")
st.markdown("Predict product ratings (1 to 5 Stars) based on customer reviews and summaries.")

if not assets_loaded:
    st.error("❌ Error loading model files (`model.pkl`, `tokenizer.pkl`, or `columns.pkl`). Please ensure all `.pkl` files exist in the working directory.")
else:
    # --- Input Form ---
    with st.form("rating_prediction_form"):
        st.subheader("📝 Enter Product Feedback")
        review_input = st.text_input("Review Headline / Short Title", placeholder="e.g., Super! or Useless product")
        summary_input = st.text_area("Detailed Review Summary", placeholder="e.g., Great cooler.. excellent air flow and worth the price.")
        
        submit_btn = st.form_submit_button("⭐ Predict Rating", type="primary", use_container_width=True)

    # --- Prediction Execution ---
    if submit_btn:
        if not review_input.strip() and not summary_input.strip():
            st.warning("⚠️ Please provide either a review headline or a summary text.")
        else:
            # 1. Combine Review and Summary as done during training
            combined_text = f"{review_input} {summary_input}".strip()
            
            # 2. Apply Custom NLP Text Preprocessing
            cleaned_text = preprocess_text(combined_text)
            
            # 3. TensorFlow Tokenizer Sequence Conversion & Padding
            try:
                # Convert text to sequence using TensorFlow Keras Tokenizer
                sequences = tokenizer.texts_to_sequences([cleaned_text])
                
                # Infer maxlen from columns or default length
                max_len = len(columns) if hasattr(columns, '__len__') else 100
                padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post')
                
                # Convert to DataFrame matching model columns if columns.pkl contains column names
                if hasattr(columns, '__len__') and not isinstance(columns, int):
                    input_data = pd.DataFrame(padded_sequence, columns=columns)
                else:
                    input_data = padded_sequence

                # 4. Model Prediction
                prediction = model.predict(input_data)[0]
                predicted_rate = int(round(float(prediction))) if isinstance(prediction, (np.number, float, int)) else prediction

                # 5. Display UI Results
                st.markdown("---")
                st.markdown("### 📊 Prediction Result")
                
                stars = "⭐" * min(max(int(predicted_rate), 1), 5) if isinstance(predicted_rate, int) else "⭐"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Rating (Rate)", value=f"{predicted_rate} / 5")
                with col2:
                    st.metric(label="Rating Stars", value=stars)

                # Contextual Alert Banner
                if isinstance(predicted_rate, int) and predicted_rate >= 4:
                    st.success("🟢 Positive Sentiment: Customers are highly satisfied with this product!")
                elif isinstance(predicted_rate, int) and predicted_rate == 3:
                    st.info("🟡 Neutral Sentiment: Average product feedback.")
                else:
                    st.error("🔴 Negative Sentiment: Critical feedback detected.")

            except Exception as err:
                st.error(f"❌ Error during inference execution: {str(err)}")