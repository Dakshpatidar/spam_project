import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and Train the Model Function
@st.cache_resource
def train_and_save_model():
    # Load the dataset with proper encoding to avoid UnicodeDecodeError
    df = pd.read_csv('spam.csv', encoding='latin1')

    # Drop unnecessary columns if present
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Convert labels to binary format
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Text preprocessing and vectorization
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['message']).toarray()
    y = df['label']

    # Train the model
    model = MultinomialNB()
    model.fit(X, y)

    # Save the vectorizer and model
    with open('tfidf_vectorizer.pkl', 'wb') as tfidf_file:
        pickle.dump(tfidf, tfidf_file)
    with open('spam_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    return tfidf, model

# Load the trained vectorizer and model or train if not available
try:
    with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
        tfidf = pickle.load(tfidf_file)
    with open('spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    tfidf, model = train_and_save_model()

# Streamlit Web App
st.title("Email/SMS Spam Classifier")

st.write("Enter the message below:")
user_input = st.text_input("Enter the message")

if st.button('Classify'):
    if user_input:
        # Preprocess and classify user input
        vector_input = tfidf.transform([user_input])
        result = model.predict(vector_input)[0]

        # Display Result
        if result == 1:
            st.error("The entered message is classified as SPAM.")
        else:
            st.success("The entered message is classified as HAM.")
    else:
        st.warning("Please enter a message to classify.")
