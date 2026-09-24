from pathlib import Path
import re

import joblib
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# Load the existing trained model and vectorizer from the project directory.
model = joblib.load(BASE_DIR / "logistic_model.pkl")
tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("text", "")

        if user_text.strip():
            cleaned_text = clean_text(user_text)
            text_tfidf = tfidf.transform([cleaned_text])
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]
            confidence = round(max(probabilities) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        user_text=user_text,
    )


if __name__ == "__main__":
    app.run(debug=True)
