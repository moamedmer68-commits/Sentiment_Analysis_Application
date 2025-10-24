import streamlit as st
from transformers import pipeline

# -------------------------------
# Streamlit page configuration
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# Load Model with Cache
# -------------------------------
@st.cache_resource
def load_model(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """Loads and caches the sentiment analysis model."""
    return pipeline("sentiment-analysis", model=model_name)

# -------------------------------
# Page Title and Description
# -------------------------------
st.title("💬 Sentiment Analysis Application")
st.markdown(
    """
    This app analyzes the **sentiment** (positive or negative) of your input text using a pre-trained
    transformer model from Hugging Face 🤗.
    """
)

# -------------------------------
# Model Selection
# -------------------------------
model_name = st.selectbox(
    "Select Model:",
    [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "finiteautomata/bertweet-base-sentiment-analysis",  # Another English model
    ],
    index=0,
    help="Choose which pre-trained model to use."
)

# -------------------------------
# Text Input
# -------------------------------
text = st.text_area(
    "Enter the text you want to analyze:",
    height=200,
    placeholder="Type or paste some text here..."
)

# -------------------------------
# Analyze Button
# -------------------------------
if st.button("Analyze Sentiment"):
    if text.strip():
        with st.spinner("Analyzing... Please wait."):
            model = load_model(model_name)
            result = model(text)[0]
            label = result["label"]
            score = result["score"]

        # -------------------------------
        # Display Results
        # -------------------------------
        st.subheader("Result:")
        if "POS" in label.upper():
            st.success(f"😊 Sentiment: {label} (confidence: {score:.2f})")
        elif "NEG" in label.upper():
            st.error(f"😠 Sentiment: {label} (confidence: {score:.2f})")
        else:
            st.info(f"😐 Sentiment: {label} (confidence: {score:.2f})")

        # Show raw output
        with st.expander("Show raw model output"):
            st.json(result)
    else:
        st.warning("⚠️ Please enter some text before analyzing.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Hugging Face Transformers.")
