import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


# Set NLTK data path to your custom folder
nltk.data.path.append(os.path.join('nltk_data'))

lemmatizer = WordNetLemmatizer()

# Load stopwords from the specified folder
stop_words = set(stopwords.words('english'))

# Load the model from the pickle file in the custom folder
def load_model(pickle_file_name):
    #pickle_file_path = os.path.join(custom_folder_path, pickle_file_name)
    with open(pickle_file_name, 'rb') as file:
        model = pickle.load(file)
    return model

def clean_text(text):
    text = str(text).lower()
    # Remove HTML tags
    text = re.sub('<[^<]+?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-English characters

    # Tokenize, remove stopwords, and lemmatize
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    text = ' '.join(lemmatized_tokens)

    return text

# Function to predict sentiment and probabilities
def predict_sentiment(model, text, vectorizer):
    # Vectorize the input text using the loaded vectorizer
    text_vectorized = vectorizer.transform([clean_text(text)])
    
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
    # Center the main title
    st.markdown("<h1 style='text-align: center;'>Sentiment Analysis App</h1>", unsafe_allow_html=True)
    
    # Input text from user
    user_input = st.text_area("Enter text for sentiment analysis", height=200)
    
    # Load the pre-trained model and vectorizer from the custom folder
    model = load_model('rfc_model.pkl')  # Path to your RandomForest pickle file
    vectorizer = load_model('tfidf_vectorizer.pkl')  # Path to your TfidfVectorizer pickle file
    
    # Center the button
    button_html = """
    <div style="display: flex; justify-content: center;">
        <button style= text-align: center; 
                       text-decoration: none; display: inline-block; font-size: 16px;">
            Predict Sentiment
        </button>
    </div>
    """
    
    if st.markdown(button_html, unsafe_allow_html=True):  # Display centered button
        if user_input:
            sorted_sentiments = predict_sentiment(model, user_input, vectorizer)
            
            # Center the sentiment probabilities title
            st.markdown("### <div style='text-align: center;'>Sentiment Probabilities:</div>", unsafe_allow_html=True)
            
            results_html = "<div style='text-align: center;'>"
            for sentiment, probability in sorted_sentiments:
                results_html += f"<p><strong>{sentiment}:</strong> {probability:.4f}</p>"
            results_html += "</div>"
            
            # Display the results in the center
            st.markdown(results_html, unsafe_allow_html=True)
        else:
            st.write(" ")

if __name__ == '__main__':
    main()
