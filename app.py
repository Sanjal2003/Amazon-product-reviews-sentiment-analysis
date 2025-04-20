# Streamlit app
import streamlit as st
import joblib
import re

def app():
    st.title("Sentiment Analysis App")
    model = joblib.load("logistic_regression_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    
    user_input = st.text_area("Enter a review:")
    # Data Cleaning Function
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    if st.button("Analyze Sentiment"):
      
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    app()