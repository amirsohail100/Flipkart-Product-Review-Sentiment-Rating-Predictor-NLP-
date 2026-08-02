import streamlit as st
import string
import pandas as pd
import numpy as np

# Safe import for joblib
try:
    import joblib
except Exception:
    joblib = None

# Safe import for TensorFlow/Keras sequence padding
pad_sequences = None
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    try:
        from keras.preprocessing.sequence import pad_sequences
    except Exception:
        pass

# --- Page Configuration ---
st.set_page_config(
    page_title="Flipkart Review Rating Predictor",
    page_icon="🛍️",
    layout="centered"
)

# --- Title and Header UI ---
st.title("🛍️ Flipkart Product Sentiment & Rating Predictor")
st.markdown("Predict product ratings (1 to 5 Stars) powered by an **Extra Trees Classifier** (83.2% Accuracy).")

# --- Dependency Checks ---
DEP_ERRORS = []

if joblib is None:
    DEP_ERRORS.append("`joblib` is not installed. Please run `pip install joblib`.")

if pad_sequences is None:
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
    except Exception:
        return str(txt)

# --- Safe Asset Loading Using Joblib ---
@st.cache_resource
def load_assets():
    model, tokenizer, columns = None, None, None
    missing_files = []
    
    if joblib is None:
        return None, None, None, ["Joblib library missing"]

    # 1. Load Model via Joblib
    try:
        model = joblib.load("model.pkl")
    except FileNotFoundError:
        missing_files.append("`model.pkl`")
    except Exception as e:
        missing_files.append(f"`model.pkl` (Error: {str(e)})")

    # 2. Load Tokenizer via Joblib
    try:
        tokenizer = joblib.load("tokenizer.pkl")
    except FileNotFoundError:
        missing_files.append("`tokenizer.pkl`")
    except Exception as e:
        missing_files.append(f"`tokenizer.pkl` (Error: {str(e)})")

    # 3. Load Columns Definition via Joblib
    try:
        columns = joblib.load("columns.pkl")
    except FileNotFoundError:
        missing_files.append("`columns.pkl`")
    except Exception as e:
        missing_files.append(f"`columns.pkl` (Error: {str(e)})")

    return model, tokenizer, columns, missing_files

model, tokenizer, columns, missing_files = load_assets()

if missing_files:
    DEP_ERRORS.append(f"Missing or invalid pickle files: {', '.join(missing_files)}")

# --- Display Dependency Warning Messages ---
if DEP_ERRORS:
    st.error("⚠️ **System Notice:** Some required dependencies or files are missing.")
    for err in DEP_ERRORS:
        st.warning(f"• {err}")

# --- Input Form ---
with st.form("rating_prediction_form"):
    st.subheader("📝 Enter Product Feedback")
    review_input = st.text_input("Review Headline / Short Title", placeholder="e.g., Super! or Useless product")
    summary_input = st.text_area("Detailed Review Summary", placeholder="e.g., Great cooler.. excellent air flow and worth the price.")
    
    is_ready = (len(DEP_ERRORS) == 0) and (model is not None) and (tokenizer is not None) and (pad_sequences is not None)
    submit_btn = st.form_submit_button("⭐ Predict Rating", type="primary", use_container_width=True, disabled=not is_ready)

if not is_ready:
    st.info("💡 **Note:** Please resolve the dependency or file issues listed above to enable real-time predictions.")

# --- Prediction Pipeline ---
if submit_btn and is_ready:
    review_str = review_input.strip() if review_input else ""
    summary_str = summary_input.strip() if summary_input else ""

    if not review_str and not summary_str:
        st.warning("⚠️ Please provide either a review headline or a summary text.")
    else:
        try:
            # 1. Create DataFrame matching training format screenshot
            input_df = pd.DataFrame([{
                "Review": review_str,
                "Summary": summary_str
            }])
            
            # 2. Apply screenshot transformations: combine and drop columns
            input_df["Text"] = input_df["Review"] + " " + input_df["Summary"]
            input_df.drop(["Review", "Summary"], axis=1, inplace=True)
            
            raw_text = input_df["Text"].iloc[0]
            
            # 3. Text Preprocessing
            cleaned_text = preprocess_text(raw_text)

            # 4. Tokenization using TensorFlow Tokenizer
            try:
                sequences = tokenizer.texts_to_sequences([cleaned_text])
            except Exception as e:
                raise RuntimeError(f"Tokenizer error: {str(e)}")

            # 5. Sequence Padding
            try:
                max_len = len(columns) if hasattr(columns, '__len__') else 100
                padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post')
            except Exception as e:
                raise RuntimeError(f"Padding error: {str(e)}")

            # 6. Column Alignment with Model Schema
            try:
                if hasattr(columns, '__len__') and not isinstance(columns, int):
                    final_input = pd.DataFrame(padded_sequence, columns=columns)
                else:
                    final_input = padded_sequence
            except Exception:
                final_input = padded_sequence

            # 7. Model Inference
            try:
                prediction = model.predict(final_input)[0]
                predicted_rate = int(round(float(prediction))) if isinstance(prediction, (np.number, float, int)) else prediction
            except Exception as e:
                raise RuntimeError(f"Inference error: {str(e)}")

            # 8. Render Results
            st.markdown("---")
            st.markdown("### 📊 Prediction Result")
            
            stars = "⭐" * min(max(int(predicted_rate), 1), 5) if isinstance(predicted_rate, int) else "⭐"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Rating (Rate)", value=f"{predicted_rate} / 5")
            with col2:
                st.metric(label="Rating Stars", value=stars)

            if isinstance(predicted_rate, int) and predicted_rate >= 4:
                st.success("🟢 Positive Sentiment: Customers are highly satisfied with this product!")
            elif isinstance(predicted_rate, int) and predicted_rate == 3:
                st.info("🟡 Neutral Sentiment: Average product feedback.")
            else:
                st.error("🔴 Negative Sentiment: Critical feedback detected.")

        except Exception as runtime_err:
            st.error(f"❌ Failed to generate prediction: {str(runtime_err)}")