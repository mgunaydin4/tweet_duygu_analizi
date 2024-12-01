import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_validate
import joblib  # Modeli kaydetmek ve yüklemek için


# Veri hazırlama fonksiyonu
def data_preparation(dataframe, tf_idfVectorizer):
    dataframe['tweet'] = dataframe['tweet'].str.lower()
    
    # Etiketleri değiştirme işlemi
    dataframe["label"] = dataframe["label"].replace(1, value="pozitif")
    dataframe["label"] = dataframe["label"].replace(-1, value="negatif")
    dataframe["label"] = dataframe["label"].replace(0, value="nötr")
    
    label_encoder = LabelEncoder()
    dataframe["label"] = label_encoder.fit_transform(dataframe["label"])
    dataframe.dropna(axis=0, inplace=True)
    
    X = tf_idfVectorizer.fit_transform(dataframe["tweet"])
    y = dataframe["label"]
    
    return X, y, label_encoder



# Logistic Regression ve Model Kaydetme fonksiyonu
def train_and_save_model(X, y, tf_idfVectorizer):
    log_model = LogisticRegression()
    log_model.fit(X, y)
    joblib.dump(log_model, 'logistic_regression_model.pkl')  # Modeli kaydet
    joblib.dump(tf_idfVectorizer, 'tf_idfVectorizer.pkl')  # TF-IDF Vectorizer'ı kaydet
    return log_model


# Tahmin fonksiyonu
def predict_tweet(tweet, log_model, tf_idfVectorizer, label_encoder):
    tweet_tfidf = tf_idfVectorizer.transform([tweet])
    prediction = log_model.predict(tweet_tfidf)[0]
    labels = label_encoder.inverse_transform([prediction])  # Label'ları tekrar çözümle
    return labels[0]


# Streamlit arayüzü
def main():
    st.title("Tweet Duygu Analizi Uygulaması")
    st.write("Bu uygulama, tweet'lerinizi pozitif, nötr veya negatif olarak sınıflandırır.")

    # Modeli yükle
    if "log_model" not in st.session_state:
        # Eğitim verisini yükle ve modeli eğit
        dataframe = pd.read_csv("tweets_labeled.csv")
        tf_idfVectorizer = TfidfVectorizer()
        X, y, label_encoder = data_preparation(dataframe, tf_idfVectorizer)
        log_model = train_and_save_model(X, y, tf_idfVectorizer)

        # Modeli ve LabelEncoder'ı kaydet
        st.session_state['log_model'] = log_model
        st.session_state['tf_idfVectorizer'] = tf_idfVectorizer
        st.session_state['label_encoder'] = label_encoder

        # Modelin başarısını hesapla
        scoring = ['accuracy', 'f1_weighted']
        cv_results = cross_validate(log_model, X, y, scoring=scoring, cv=10)
        st.session_state['accuracy'] = cv_results['test_accuracy'].mean()
        st.session_state['f1'] = cv_results['test_f1_weighted'].mean()
        st.success("Model hazır!")

    # Kullanıcıdan tweet girişi
    user_tweet = st.text_input("Bir tweet yazın:")

    if st.button("Tahmin Et"):
        if user_tweet.strip() != "":
            log_model = st.session_state['log_model']
            tf_idfVectorizer = st.session_state['tf_idfVectorizer']
            label_encoder = st.session_state['label_encoder']

            result = predict_tweet(user_tweet, log_model, tf_idfVectorizer, label_encoder)
            st.write(f"Tahmin edilen duygu: **{result}**")
            st.success(
                f"**Accuracy (Mean): %** {(st.session_state['accuracy'] * 100).round(2)}\n\n**F1 Score (Mean): %** {(st.session_state['f1'] * 100).round(2)}")
        else:
            st.warning("Lütfen bir tweet giriniz.")


if __name__ == "__main__":
    main()
