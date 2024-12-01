import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_validate

# Veri hazırlama fonksiyonu
def data_preparation(dataframe, tf_idfVectorizer):
    dataframe['tweet'] = dataframe['tweet'].str.lower()
    dataframe["label"].replace(1, value="pozitif", inplace=True)
    dataframe["label"].replace(-1, value="negatif", inplace=True)
    dataframe["label"].replace(0, value="nötr", inplace=True)
    dataframe["label"] = LabelEncoder().fit_transform(dataframe["label"])
    dataframe.dropna(axis=0, inplace=True)
    X = tf_idfVectorizer.fit_transform(dataframe["tweet"])
    y = dataframe["label"]
    return X, y

# Logistic Regression fonksiyonu
def logistic_regression(X, y):
    log_model = LogisticRegression(max_iter=10000).fit(X, y)
    scoring = ['accuracy', 'f1_weighted']
    cv_results = cross_validate(log_model, X, y, scoring=scoring, cv=10)
    accuracy = cv_results['test_accuracy'].mean()
    f1 = cv_results['test_f1_weighted'].mean()
    return log_model, accuracy, f1

# Tahmin fonksiyonu
def predict_tweet(tweet, log_model, tf_idfVectorizer):
    tweet_tfidf = tf_idfVectorizer.transform([tweet])
    prediction = log_model.predict(tweet_tfidf)[0]
    labels = {0: "negatif", 1: "nötr", 2: "pozitif"}
    return labels[prediction]

# Streamlit arayüzü
def main():
    st.title("Tweet Duygu Analizi Uygulaması")
    st.write("Bu uygulama, tweet'lerinizi pozitif, nötr veya negatif olarak sınıflandırır.")

    # Modeli hazırlamak için eğitim verisi yüklenir
    st.write("Model hazırlanıyor...")
    dataframe = pd.read_csv("Miuul DSML16/7-NLP/Graduate Project/SentimentyBot/tweets_labeled.csv")
    tf_idfVectorizer = TfidfVectorizer()
    X, y = data_preparation(dataframe, tf_idfVectorizer)
    log_model, accuracy, f1 = logistic_regression(X, y)
    st.success("Model hazır!")



    # Kullanıcıdan tweet girişi
    user_tweet = st.text_input("Bir tweet yazın:")

    if st.button("Tahmin Et"):
        if user_tweet.strip() != "":
            result = predict_tweet(user_tweet, log_model, tf_idfVectorizer)
            st.write(f"Tahmin edilen duygu: **{result}**")
            st.success(f"**Accuracy (Mean): %** {(accuracy * 100).round(2)}\n\n**F1 Score (Mean): %** {(f1 * 100).round(2)}")
        else:
            st.warning("Lütfen bir tweet giriniz.")

if __name__ == "__main__":
    main()
