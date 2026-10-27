
import streamlit as st
import joblib

st.title("Fake News Detection System")
st.write("Enter a news article to check whether it is Real or Fake.")

model = joblib.load("fake_news_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

news = st.text_area("Enter news text here:")

if st.button("Check News"):
    if news.strip():
        text = vectorizer.transform([news])
        prediction = model.predict(text)[0]

        if prediction == 1:
            st.success("This news is likely REAL.")
        else:
            st.error("This news is likely FAKE.")
    else:
        st.warning("Please enter some news text.")
