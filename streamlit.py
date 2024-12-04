
import streamlit as st
import joblib
import pandas as pd

# Modelleri yükleme
log_model = joblib.load("log_model.pkl")
tf_idf_vectorizer = joblib.load("tf_idf_vectorizer.pkl")


st.title("Tweet Duygu Analizi Uygulaması")
st.subheader("Bu uygulama, tweet'lerinizi pozitif, nötr veya negatif olarak sınıflandırır.")

# Kullanıcıdan tweet girişi
user_tweet = st.text_input("Tweetinizi buraya yazın:")

if st.button("Tahmin Et"):
    if user_tweet.strip() == "":
        st.warning("Lütfen bir tweet girin.")
    else:
        # TF-IDF dönüşümü
        tweet_tfidf = tf_idf_vectorizer.transform([user_tweet.lower()])
        prediction = log_model.predict(tweet_tfidf)[0]
        # Tahmin sonucu
        if prediction == 0:
            sentiment = "Negatif"
        elif prediction == 1:
            sentiment = "Nötr"
        elif prediction == 2:
            sentiment = "Pozitif"

        st.success(f"Tahmin edilen duygu durumu: {sentiment}")
        st.info(f"**Accuracy (Mean): %** 67.03\n\n**F1 Score (Mean): %** 60.12")


def warning(param):
    return None
