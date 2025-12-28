import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # remove spacial character
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Email/SMS Spam Classifier")

input_text = st.text_area("Enter The Message")

if st.button("Predict"):

 #1.preprocess
 transformed_text = transform_text(input_text)
 #2.vectorize
 vectorized_text = tfidf.transform([transformed_text])
 #3.predict
 result = model.predict(vectorized_text)
 #4.display
 if result == 1:
    st.header("SPAM")
 else:
    st.header("NOT SPAM")

