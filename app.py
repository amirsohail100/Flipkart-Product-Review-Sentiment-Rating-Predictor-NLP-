import streamlit as st
import pickle
import string
import pandas as pd
import numpy as np

# --- Import Fallback for TensorFlow/Keras ---
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    try:
        from keras.preprocessing.sequence import pad_sequences
    except Exception as e:
        pad_sequences = None

# --- Page Configuration ---
st.set_page_config(
    page_title="Flipkart Review Rating Predictor",
    page_icon="🛍️",
    layout="centered"
)

# --- NLP Preprocessing Functions ---
def remove_punc(txt):
    try:
        return txt.translate(str.maketrans('', '', string.punctuation))
    except Exception:
        return str(txt)

def tolower(txt):
    try:
        return txt.lower()
    except Exception:
        return str(txt)

def remove_num(txt):
    try:
        new = ""
        for i in txt:
            if not i.isdigit():
                new += i
        return new
    except Exception:
        return str(txt)

def remove_emoj(txt):
    try:
        new = ""
        for i in txt:
            if i.isascii():
                new += i
        return new
    except Exception:
        return str(txt)

def preprocess_text(txt):
    try:
        txt = remove_punc(txt)
        txt = tolower(txt)
        txt = remove_num(txt)
        txt = remove_emoj(txt)
        return txt
    except Exception as e:
        st.warning(f"Warning during text preprocessing: {str(e)}")
        return str(txt)

# --- Load Models & Assets Safely ---
@st.cache_resource
def load_assets():
    model, tokenizer, columns = None, None, None
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("columns.pkl", "rb") as f:
            columns = pickle.load(f)
        return model, tokenizer, columns, None
    except FileNotFoundError as fnf_err:
        return None, None, None, f"File missing error: {str(fnf_err)}"
    except Exception as e:
        return None, None, None, f"Failed to load assets: {str(e)}"

model, tokenizer, columns, load_error = load_assets()

# --- UI Layout ---
st.title("🛍️ Flipkart Product Sentiment & Rating Predictor")
st.markdown("Predict product ratings (1 to 5 Stars) powered by an **Extra Trees Classifier** (83.2% Accuracy).")

if load_error or pad_sequences is None:
    if pad_sequences is None:
        st.error("❌ Critical Error: TensorFlow/Keras is not installed or failed to import `pad_sequences`.")
    if load_error:
        st.error(f"❌ Asset Loading Error: {load_error}")
else:
    # --- Input Form ---
    with st.form("rating_prediction_form"):
        st.subheader("📝 Enter Product Feedback")
        review_input = st.text_input("Review Headline / Short Title", placeholder="e.g., Super! or Useless product")
        summary_input = st.text_area("Detailed Review Summary", placeholder="e.g., Great cooler.. excellent air flow and worth the price.")
        
        submit_btn = st.form_submit_button("⭐ Predict Rating", type="primary", use_container_width=True)

    # --- Prediction Execution ---
    if submit_btn:
        review_str = review_input.strip() if review_input else ""
        summary_str = summary_input.strip() if summary_input else ""

        if not review_str and not summary_str:
            st.warning("⚠️ Please provide either a review headline or a summary text.")
        else:
            try:
                # 1. Combine Review and Summary
                combined_text = f"{review_str} {summary_str}".strip()
                
                # 2. Preprocess Text
                cleaned_text = preprocess_text(combined_text)
                
                # 3. Tokenize Sequence with Exception Handling
                try:
                    sequences = tokenizer.texts_to_sequences([cleaned_text])
                except Exception as tok_err:
                    raise RuntimeError(f"Tokenizer conversion failed: {str(tok_err)}")

                # 4. Infer Max Length & Pad Sequence
                try:
                    max_len = len(columns) if hasattr(columns, '__len__') else 100
                    padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post')
                except Exception as pad_err:
                    raise RuntimeError(f"Sequence padding failed: {str(pad_err)}")

                # 5. Column Formatting
                try:
                    if hasattr(columns, '__len__') and not isinstance(columns, int):
                        input_data = pd.DataFrame(padded_sequence, columns=columns)
                    else:
                        input_data = padded_sequence
                except Exception:
                    input_data = padded_sequence

                # 6. Prediction Execution
                try:
                    prediction = model.predict(input_data)[0]
                    predicted_rate = int(round(float(prediction))) if isinstance(prediction, (np.number, float, int)) else prediction
                except Exception as pred_err:
                    raise RuntimeError(f"Model prediction failed: {str(pred_err)}")

                # 7. Render UI Output
                st.markdown("---")
                st.markdown("### 📊 Prediction Result")
                
                stars = "⭐" * min(max(int(predicted_rate), 1), 5) if isinstance(predicted_rate, int) else "⭐"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Rating (Rate)", value=f"{predicted_rate} / 5")
                with col2:
                    st.metric(label="Rating Stars", value=stars)

                # Contextual Sentiment Banners
                if isinstance(predicted_rate, int) and predicted_rate >= 4:
                    st.success("🟢 Positive Sentiment: Customers are highly satisfied with this product!")
                elif isinstance(predicted_rate, int) and predicted_rate == 3:
                    st.info("🟡 Neutral Sentiment: Average product feedback.")
                else:
                    st.error("🔴 Negative Sentiment: Critical feedback detected.")

            except Exception as main_err:
                st.error(f"❌ An error occurred during prediction: {str(main_err)}")