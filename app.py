import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load model and tokenizer
# (compile=False avoids unnecessary training/metrics warnings during inference)
model = load_model('model/model.h5', compile=False)
tokenizer = joblib.load('model/tokenizer.pkl')

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "ℹ️ About", "🧠 Model Info", "📝 Instructions", "⚠️ Disclaimer"]
)

# Prediction helper
def predictive_system(review: str) -> float:
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=200)
    prediction = model.predict(padded, verbose=0)
    return float(prediction[0][0])

# --- HOME ---
if page == "🏠 Home":
    st.title("🎬 IMDB Movie Review Sentiment Analysis")
    st.markdown("""
    Welcome to the **IMDB Movie Review Sentiment Analysis App**!

    This app uses a **Deep Learning** model to classify movie reviews as:
    - ✅ **Positive** (the reviewer enjoyed the movie)
    - ❌ **Negative** (the reviewer did not enjoy the movie)

    Enter a review below to see:
    - The predicted **sentiment**
    - A **confidence score**
    """)

    user_input = st.text_area("✍️ Enter a movie review below:")

    if st.button("🔍 Classify"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review before classifying.")
        else:
            prob = predictive_system(user_input)

            st.subheader("📊 Prediction Result")
            positive = prob > 0.45
            conf = prob if positive else (1 - prob)
            conf_pct = int(round(conf * 100))

            if positive:
                st.success(f"✅ Positive Review ({conf_pct:.0f}% confidence)")
            else:
                st.error(f"❌ Negative Review ({conf_pct:.0f}% confidence)")

            # Confidence bar (0–100)
            st.progress(conf_pct)

            # Professional summary (neutral wording)
            st.markdown(f"""
            ### 📝 Analysis Summary
            - **Classification:** {'Positive' if positive else 'Negative'}
            - **Confidence Level:** {conf_pct:.0f}%
            - **Threshold Used:** 0.45 (scores above this are classified as Positive)
            """)

            st.markdown("""
            **Why this matters?**  
            Sentiment analysis can help:
            - 🎥 Movie studios understand audience reception  
            - 🛒 Businesses gauge customer feedback  
            - 💬 Social platforms filter toxic content  
            """)

    # Small reminder linking to full disclaimer
    st.caption("🔎 For limitations and usage notes, see the **Disclaimer** page in the sidebar.")

# --- ABOUT ---
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    The **IMDB Movie Review Sentiment Analysis App** showcases how
    **Natural Language Processing (NLP)** and **Deep Learning** can turn free-form text into measurable insights.

    ### Why Sentiment Analysis?
    People write reviews everywhere—movies, products, services. Instead of reading everything,
    automated sentiment analysis summarizes the **overall tone** quickly.

    ### Features
    - Classifies text as **Positive** or **Negative**
    - Provides a **confidence level** and **visual indicator**
    - Clear, professional result presentation (no subjective wording)
    - Pages explaining how the model works and how to use it

    ### Example Uses
    - 🎬 Film industry: gauge audience response
    - 🛍️ E-commerce: summarize product feedback at scale
    - 📱 Social media: monitor sentiment on topics or brands

    This demo focuses on movie reviews to illustrate how AI can interpret written opinions.
    """)

# --- MODEL INFO ---
elif page == "🧠 Model Info":
    st.title("🧠 Model Information")
    st.markdown("""
    The backbone is a **Recurrent Neural Network (RNN)** using an **LSTM** (Long Short-Term Memory) layer.

    ### Model Details
    - **Architecture:** Embedding → LSTM → Dense (sigmoid)
    - **Embedding Dimension:** 128
    - **Input Length:** 200 tokens (reviews are padded/truncated)
    - **Training Data:** IMDB movie reviews (binary sentiment)
    - **Tokenizer:** Top 5,000 most frequent tokens

    ### How It Works
    1. Text is **tokenized** to integer IDs and **padded** to length 200  
    2. The **Embedding** layer maps tokens to dense vectors  
    3. The **LSTM** captures context across the sequence  
    4. A final **sigmoid** outputs a score in \[0,1] (higher → more positive)

    ### Typical Limitations
    - Sarcasm and irony can be misclassified  
    - Mixed/neutral phrasing may yield uncertain scores  
    - Best performance on English; domain shifts reduce accuracy
    """)

# --- INSTRUCTIONS ---
elif page == "📝 Instructions":
    st.title("📝 How to Use the App")
    st.markdown("""
    1. Open **Home** in the sidebar.  
    2. Paste or type a movie review.  
    3. Click **Classify** to see the label and confidence.

    ### Example Inputs
    - **Positive:** “Absolutely fantastic acting and a compelling story.”  
    - **Negative:** “Predictable plot and weak performances.”  
    - **Mixed:** “Great start but the ending fell flat.”

    ### Tips
    - Use full sentences for more reliable results  
    - English input works best (trained on IMDB English reviews)  
    - Very short texts may yield lower confidence
    """)

# --- DISCLAIMER ---
elif page == "⚠️ Disclaimer":
    st.title("⚠️ Disclaimer")
    st.warning("""
    **Educational Use Only.**  
    This application is intended for learning and demonstration purposes.
    It provides automated sentiment classification using a pretrained model and may be inaccurate or incomplete.
    """)

    st.markdown("""
    ### Important Notes
    - Results are **not guaranteed** to be correct in all cases.  
    - The model can be sensitive to phrasing, sarcasm, and domain shifts.  
    - Do **not** use these outputs as the sole basis for decisions with legal, medical, financial, or safety implications.  
    - Inputs are processed to generate predictions; avoid submitting sensitive or personal data.

    By using this app, you agree that it is provided **“as is”**, without warranties of any kind,
    and solely for **educational and experimental** purposes.
    """)
