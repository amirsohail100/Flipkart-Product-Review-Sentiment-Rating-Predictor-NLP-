import streamlit as st
import pickle
import string
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Flipkart Review Rating Predictor",
    page_icon="🛍️",
    layout="centered"
)

# --- NLP Preprocessing Functions (As defined in notebook) ---
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
st.markdown("Predict product rating (1 to 5 Stars) based on customer reviews and summaries.")

if not assets_loaded:
    st.error("❌ Error loading model files (`model.pkl`, `tokenizer.pkl`, or `columns.pkl`). Please make sure all `.pkl` files are in the working directory.")
else:
    # --- Input Form ---
    with st.form("rating_prediction_form"):
        st.subheader("📝 Enter Product Feedback")
        review_input = st.text_input("Review Headline / Short Title", placeholder="e.g., Useless product or Super!")
        summary_input = st.text_area("Detailed Review Summary", placeholder="e.g., Great cooler.. excellent air flow and worth the price.")
        
        submit_btn = st.form_submit_button("⭐ Predict Rating", type="primary", use_container_width=True)

    # --- Prediction Execution ---
    if submit_btn:
        if not review_input.strip() and not summary_input.strip():
            st.warning("⚠️ Please provide either a review title or a summary text.")
        else:
            # 1. Combine Review and Summary as done in training
            combined_text = f"{review_input} {summary_input}".strip()
            
            # 2. Apply Custom NLP Text Preprocessing
            cleaned_text = preprocess_text(combined_text)
            
            # 3. Vectorize / Tokenize Text
            try:
                text_vector = tokenizer.transform([cleaned_text])
                
                # Align columns if columns.pkl specifies feature matrix alignment
                if hasattr(text_vector, "toarray"):
                    text_df = pd.DataFrame(text_vector.toarray())
                else:
                    text_df = pd.DataFrame(text_vector)

                # 4. Model Prediction
                prediction = model.predict(text_df)[0]
                
                # Handle array or scalar prediction
                predicted_rate = int(round(float(prediction))) if isinstance(prediction, (np.number, float, int)) else prediction

                # 5. Display UI Results
                st.markdown("---")
                st.markdown("### 📊 Prediction Result")
                
                # Rating Star Badge Display
                stars = "⭐" * min(max(int(predicted_rate), 1), 5) if isinstance(predicted_rate, int) else "⭐"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Rating (Rate)", value=f"{predicted_rate} / 5")
                with col2:
                    st.metric(label="Rating Stars", value=stars)

                # Contextual Alert Banner
                if isinstance(predicted_rate, int) and predicted_rate >= 4:
                    st.success("🟢 Positive Sentiment: Users are highly satisfied with this product!")
                elif isinstance(predicted_rate, int) and predicted_rate == 3:
                    st.info("🟡 Neutral Sentiment: Average product feedback.")
                else:
                    st.error("🔴 Negative Sentiment: Critical feedback detected.")

            except Exception as err:
                st.error(f"❌ Error during inference execution: {str(err)}")