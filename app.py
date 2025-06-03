import streamlit as st
import pickle
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import nltk
import os

# Define local nltk data directory relative to this file
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Add the local nltk_data path BEFORE downloading or loading
nltk.data.path.append(nltk_data_dir)

# Download punkt and stopwords to the local directory only if not present
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_dir)

download_nltk_resources()

# Initialize stemmer
stemmer = PorterStemmer()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    filtered_tokens = []
    for word in tokens:
        if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
            stemmed = stemmer.stem(word)
            filtered_tokens.append(stemmed)

    return " ".join(filtered_tokens)

# Load vectorizer and model
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
classifier = pickle.load(open('spam_classifier.pkl', 'rb'))

# UI setup
st.title("📨 Email / SMS Spam Classifier")
st.markdown("Built with 🧠 Machine Learning and NLP")

user_input = st.text_area("✉️ Enter your message below:")

if st.button("Classify"):
    # Preprocess input
    processed_input = clean_text(user_input)
    vectorized_text = vectorizer.transform([processed_input])
    prediction = classifier.predict(vectorized_text)[0]
    proba = classifier.predict_proba(vectorized_text)[0]

    # Display result
    if prediction == 1:
        st.markdown("### 🚨 This message is **Spam**")
    else:
        st.markdown("### ✅ This message is **Not Spam**")

    # Show prediction probabilities as bar chart
    st.markdown("#### 📊 Prediction Confidence (Bar Chart)")
    labels = ['Not Spam', 'Spam']
    fig, ax = plt.subplots()
    ax.bar(labels, proba, color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # Show pie chart
    st.markdown("#### 🥧 Prediction Confidence (Pie Chart)")
    fig2, ax2 = plt.subplots()
    ax2.pie(proba, labels=labels, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

    # Message statistics
    st.markdown("#### 📝 Message Statistics")
    st.write(f"🔹 Original word count: {len(user_input.split())}")
    st.write(f"🔹 After preprocessing: {len(processed_input.split())}")

    # Word Cloud
    st.markdown("#### ☁️ Word Cloud of Message")
    wc = WordCloud(width=600, height=300, background_color='white').generate(processed_input)
    fig3, ax3 = plt.subplots()
    ax3.imshow(wc, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)
