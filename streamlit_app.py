import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model_dir = './bert_model'  # Adjust the path to where your model is saved
tokenizer = BertTokenizer.from_pretrained(f"{model_dir}/tokenizer")
model = BertForSequenceClassification.from_pretrained(model_dir)

# Define prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction].item()
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

# Streamlit UI
st.title("Movie Review Sentiment Analyzer")
st.write("Analyze the sentiment of your movie reviews using a BERT model.")

# Text input
review_text = st.text_area("Enter a movie review:", "")

if st.button("Analyze Sentiment"):
    if review_text.strip():
        sentiment, confidence = predict_sentiment(review_text)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.write("Please enter a valid review.")