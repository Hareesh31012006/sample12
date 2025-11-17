import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import torch
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# ---------------------------------------
# CONFIG
# ---------------------------------------
ALPHAVANTAGE_KEY = "YOUR_ALPHA_VANTAGE_KEY"
MODEL_PATH = "daily_bilstm_model.h5"

# ---------------------------------------
# LOAD FINBERT
# ---------------------------------------
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, finbert = load_finbert()


# ---------------------------------------------------------
# 1️⃣ FETCH NEWS FROM ALPHAVANTAGE (NO TWITTER, NO NEWSAPI)
# ---------------------------------------------------------
def fetch_news(ticker):
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHAVANTAGE_KEY}"
    )
    r = requests.get(url).json()

    if "feed" not in r:
        return []

    return [item["title"] for item in r["feed"]][:10]


# ---------------------------------------------------------
# 2️⃣ SENTIMENT ANALYSIS MODULE (TextBlob + FinBERT)
# ---------------------------------------------------------
def textblob_sentiment(texts):
    if not texts:
        return 0, 0
    pol = [TextBlob(t).sentiment.polarity for t in texts]
    sub = [TextBlob(t).sentiment.subjectivity for t in texts]
    return sum(pol)/len(pol), sum(sub)/len(sub)


def finbert_sentiment(texts):
    if not texts:
        return 0
    scores = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        logits = finbert(**inputs).logits
        prob = torch.softmax(logits, dim=1).detach().numpy()[0]
        score = prob[2] - prob[0]
        scores.append(score)
    return sum(scores) / len(scores)


def get_sentiment_vector(ticker):
    news = fetch_news(ticker)

    polarity, subjectivity = textblob_sentiment(news)
    finbert_score = finbert_sentiment(news)

    return polarity, subjectivity, finbert_score, news


# ---------------------------------------------------------
# 3️⃣ LOAD DAILY STOCK DATA + ADD SENTIMENT
# ---------------------------------------------------------
def load_stock_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    df.dropna(inplace=True)

    pol, sub, finbert_score, news = get_sentiment_vector(ticker)
    df["polarity"] = pol
    df["subjectivity"] = sub
    df["finbert"] = finbert_score
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    return df, news


# ---------------------------------------------------------
# 4️⃣ CREATE SEQUENCES FOR LSTM
# ---------------------------------------------------------
def create_sequences(df, window=60):
    X, y = [], []
    arr = df.values
    for i in range(len(arr) - window):
        X.append(arr[i:i + window])
        y.append(arr[i + window][3])  # Close
    return np.array(X), np.array(y)


# ---------------------------------------------------------
# 5️⃣ TRAIN BI-LSTM MODEL
# ---------------------------------------------------------
def train_model(df):
    features = [
        "Open", "High", "Low", "Close", "Volume", "Return",
        "polarity", "subjectivity", "finbert"
    ]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = create_sequences(pd.DataFrame(scaled))

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X, y,
        validation_split=0.2,
        epochs=15,
        batch_size=32,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
    )

    model.save(MODEL_PATH)

    return scaler


# ---------------------------------------------------------
# 6️⃣ MAKE NEXT-DAY PREDICTION
# ---------------------------------------------------------
def recommend_signal(pred, last_close, sentiment):
    pct = (pred - last_close) / last_close * 100

    if pct > 1 and sentiment > 0.1:
        return "BUY"
    if pct < -1 and sentiment < -0.1:
        return "SELL"
    return "HOLD"


def predict_next(df, scaler):
    model = load_model(MODEL_PATH)

    scaled = scaler.transform(df.values)[-60:].reshape(1, 60, df.shape[1])
    pred = model.predict(scaled)[0][0]

    last_close = df["Close"].iloc[-1]
    sentiment = df["finbert"].iloc[-1]

    signal = recommend_signal(pred, last_close, sentiment)

    return pred, sentiment, signal


# ==========================================================
# 🟢 STREAMLIT UI
# ==========================================================
st.title("📈 AI Stock Predictor (Daily BI-LSTM + Sentiment from AlphaVantage)")
st.caption("Uses FinBERT + TextBlob + BI-LSTM + AlphaVantage news sentiment")

ticker = st.text_input("Enter stock ticker:", "AAPL")

if st.button("Analyze"):
    with st.spinner("Fetching data & analyzing sentiment..."):
        df, news = load_stock_data(ticker)

    st.subheader("📰 Latest News Headlines")
    for n in news:
        st.write("- " + n)

    with st.spinner("Training BI-LSTM Model (Daily)..."):
        scaler = train_model(df)

    pred, sentiment, signal = predict_next(df, scaler)

    st.subheader("📊 Prediction Results")
    st.metric("Next-Day Close Prediction", f"${pred:.2f}")
    st.metric("FinBERT Sentiment", f"{sentiment:.3f}")
    st.success(f"📌 Recommendation: **{signal}**")

    st.subheader("📉 Stock Price History")
    st.line_chart(df["Close"])
