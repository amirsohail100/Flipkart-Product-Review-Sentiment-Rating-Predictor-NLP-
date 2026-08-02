import streamlit as st
import string
import pickle
import pandas as pd
import numpy as np

# --- Page Configuration (Always runs first) ---
st.set_page_config(
    page_title="Flipkart Review Rating Predictor",
    page_icon="🛍️",
    layout="centered"
)

# --- Title and Header UI (Always renders even if dependencies fail) ---
st.title("🛍️ Flipkart Product Sentiment & Rating Predictor")
st.markdown("Predict product ratings (1 to 5 Stars) powered by an **Extra Trees Classifier** (83.2% Accuracy).")

# --- Safe Dependency Checks ---
DEP_ERRORS = []

# Check TensorFlow / Keras Dependency
pad_sequences = None
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    try:
        from keras.preprocessing.sequence import pad_sequences
    except Exception as e:
        DEP_ERRORS.append("TensorFlow/Keras is not installed or failed to import `pad_sequences`.")

# --- Preprocessing Functions ---
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
        return "".join([i for i in str(txt) if not i.isdigit()])
    except Exception:
        return str(txt)

def remove_emoj(txt):
    try:
        return "".join([i for i in str(txt) if i.isascii()])
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
        return str(txt)

# --- Safe Asset Loading ---
@st.cache_resource
def load_assets():
    model, tokenizer, columns = None, None, None
    missing_files = []
    
    # 1. Load Model
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        missing_files.append("`model.pkl`")
    except Exception as e:
        missing_files.append(f"`model.pkl` (Corrupt: {str(e)})")

    # 2. Load Tokenizer
    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        missing_files.append("`tokenizer.pkl`")
    except Exception as e:
        missing_files.append(f"`tokenizer.pkl` (Corrupt: {str(e)})")

    # 3. Load Columns Definition
    try:
        with open("columns.pkl", "rb") as f:
            columns = pickle.load(f)
    except FileNotFoundError:
        missing_files.append("`columns.pkl`")
    except Exception as e:
        missing_files.append(f"`columns.pkl` (Corrupt: {str(e)})")

    return model, tokenizer, columns, missing_files

model, tokenizer, columns, missing_files = load_assets()

# Collect all potential loading errors
if missing_files:
    DEP_ERRORS.append(f"Missing or invalid pickle files: {', '.join(missing_files)}")

# --- Show UI Notifications for Errors (UI keeps running) ---
if DEP_ERRORS:
    st.error("⚠️ **System Degradation Notice:** Some required features or dependencies are unavailable.")
    for err in DEP_ERRORS:
        st.warning(f"• {err}")

# --- Input Form (Always visible) ---
with st.form("rating_prediction_form"):
    st.subheader("📝 Enter Product Feedback")
    review_input = st.text_input("Review Headline / Short Title", placeholder="e.g., Super! or Useless product")
    summary_input = st.text_area("Detailed Review Summary", placeholder="e.g., Great cooler.. excellent air flow and worth the price.")
    
    # Disable prediction button gracefully if dependencies/files are missing
    is_ready = (len(DEP_ERRORS) == 0) and (model is not None) and (tokenizer is not None) and (pad_sequences is not None)
    submit_btn = st.form_submit_button("⭐ Predict Rating", type="primary", use_container_width=True, disabled=not is_ready)

if not is_ready:
    st.info("💡 **Note:** Please resolve the dependency or missing file issues listed above to enable real-time predictions.")

# --- Prediction Execution ---
if submit_btn and is_ready:
    review_str = review_input.strip() if review_input else ""
    summary_str = summary_input.strip() if summary_input else ""

    if not review_str and not summary_str:
        st.warning("⚠️ Please provide either a review headline or a summary text.")
    else:
        try:
            # Step 1: Preprocess Text
            combined_text = f"{review_str} {summary_str}".strip()
            cleaned_text = preprocess_text(combined_text)

            # Step 2: Tokenize Sequence
            try:
                sequences = tokenizer.texts_to_sequences([cleaned_text])
            except Exception as e:
                raise RuntimeError(f"Tokenizer error: {str(e)}")

            # Step 3: Sequence Padding
            try:
                max_len = len(columns) if hasattr(columns, '__len__') else 100
                padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post')
            except Exception as e:
                raise RuntimeError(f"Padding error: {str(e)}")

            # Step 4: DataFrame Mapping
            try:
                if hasattr(columns, '__len__') and not isinstance(columns, int):
                    input_data = pd.DataFrame(padded_sequence, columns=columns)
                else:
                    input_data = padded_sequence
            except Exception:
                input_data = padded_sequence

            # Step 5: Model Prediction
            try:
                prediction = model.predict(input_data)[0]
                predicted_rate = int(round(float(prediction))) if isinstance(prediction, (np.number, float, int)) else prediction
            except Exception as e:
                raise RuntimeError(f"Inference error: {str(e)}")

            # Step 6: Render Prediction Output UI
            st.markdown("---")
            st.markdown("### 📊 Prediction Result")
            
            stars = "⭐" * min(max(int(predicted_rate), 1), 5) if isinstance(predicted_rate, int) else "⭐"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Rating (Rate)", value=f"{predicted_rate} / 5")
            with col2:
                st.metric(label="Rating Stars", value=stars)

            # Contextual Sentiments
            if isinstance(predicted_rate, int) and predicted_rate >= 4:
                st.success("🟢 Positive Sentiment: Customers are highly satisfied with this product!")
            elif isinstance(predicted_rate, int) and predicted_rate == 3:
                st.info("🟡 Neutral Sentiment: Average product feedback.")
            else:
                st.error("🔴 Negative Sentiment: Critical feedback detected.")

        except Exception as runtime_err:
            st.error(f"❌ Failed to generate prediction: {str(runtime_err)}")