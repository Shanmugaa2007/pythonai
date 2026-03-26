from fastapi import FastAPI
import pandas as pd
import joblib
import os
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_url(text):
    return bool(re.search(r'https?://|www\.', text))

def extract_url_features(url):
    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('/'),
        1 if 'https' in url else 0,   
        1 if '@' in url else 0,      
        1 if 'login' in url else 0,   
        1 if 'bank' in url else 0     
    ]


def train_models():
    print("Training models...")

    url_df = pd.read_csv("./app/data/PhishingUrlData.csv")

    url_df.columns = url_df.columns.str.lower()

    url_df = url_df.dropna()
    url_df = url_df[url_df["url"].notnull()]

    X_url = [extract_url_features(u) for u in url_df["url"]]
    y_url = url_df["label"].map({0:1,1:0})

    url_model = RandomForestClassifier(n_estimators=100)
    url_model.fit(X_url, y_url)

    joblib.dump(url_model, "url_model.pkl")

    email_df = pd.read_csv("./app/data/PhishingEmailData.csv")

    email_df.columns = email_df.columns.str.lower()

    email_df = email_df.dropna()
    email_df = email_df[email_df["email_content"].notnull()]

    vectorizer = TfidfVectorizer(stop_words="english")
    X_email = vectorizer.fit_transform(email_df["email_content"])
    y_email = email_df["label"]

    email_model = LogisticRegression(max_iter=1000)
    email_model.fit(X_email, y_email)

    joblib.dump(email_model, "email_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("Models trained successfully ")

models_exist = (
    os.path.exists("url_model.pkl") and
    os.path.exists("email_model.pkl") and
    os.path.exists("vectorizer.pkl")
)

if not models_exist:
    train_models()

url_model = joblib.load("url_model.pkl")
email_model = joblib.load("email_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.get("/")
def root():
    return {"Python server is running"}

@app.post("/check")
def check(data: dict):
    input_text = data.get("message", "")
    
    words = ["login","bank","verify","account","update","secure","password","signin","validation","authenticate"]
    if not input_text:
        return {"error": "No input provided"}

    if is_url(input_text):
        features = extract_url_features(input_text)
        probs = url_model.predict_proba([features])[0]
        input_type = "URL"
        index=1
        for word in words:
            if word in input_text:
                index=0
                print(word)
                break
    
    else:
        vec = vectorizer.transform([input_text])
        probs = email_model.predict_proba(vec)[0]
        input_type = "Email"
        index = 0
    
    phishing_prob = probs[index]
    risk_score = round(phishing_prob * 100, 2)

    if risk_score < 45:
        result = "Safe"
    elif risk_score < 75:
        result = "Suspicious"
    else:
        result = "Phishing"
    return {
        "type": input_type,
        "result": result,
        "risk_score": risk_score
    }
