import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import xgboost as xgb

# Load the XGBoost model from the pickle file
def load_model(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict sentiment and probabilities
def predict_sentiment(model, text, vectorizer):
    # Vectorize the input text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])
    
    # Predict the sentiment probabilities
    pred_proba = model.predict_proba(text_vectorized)[0]
    
    # Define sentiment labels (Assuming 0: Negative, 1: Neutral, 2: Positive)
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    
    # Create a dictionary of sentiment labels and their probabilities
    sentiment_probabilities = {sentiment_labels[i]: pred_proba[i] for i in range(len(sentiment_labels))}
    
    # Sort sentiment probabilities in descending order
    sorted_sentiments = sorted(sentiment_probabilities.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_sentiments

# Streamlit app UI
def main():
    st.title('Sentiment Analysis App')
    
    # Input text from user
    user_input = st.text_area("Enter text for sentiment analysis", height=200)
    
    # Load the pre-trained model and vectorizer
    model = load_model('xgb_model.pkl')  # Path to your XGBoost pickle file
    vectorizer = load_model('vectorizer.pkl')  # Path to your vectorizer pickle file (e.g., TfidfVectorizer)
    
    # Predict sentiment when button is clicked
    if st.button('Predict Sentiment'):
        if user_input:
            sorted_sentiments = predict_sentiment(model, user_input, vectorizer)
            
            # Display sentiment results in descending order of probabilities
            st.write("### Sentiment Probabilities:")
            for sentiment, probability in sorted_sentiments:
                st.write(f"**{sentiment}**: {probability:.4f}")
        else:
            st.write("Please enter some text for sentiment analysis.")

if __name__ == '__main__':
    main()
