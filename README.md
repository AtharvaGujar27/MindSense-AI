# MindSense AI

MindSense AI is a web application that uses natural language processing to classify text with a trained machine learning model. Enter a short piece of text to see the predicted category and the model’s confidence score.

> **Disclaimer:** This project is for educational purposes only. It is not a mental health diagnostic tool and should not replace guidance from a qualified professional.

## Features

- Classifies text using a trained Logistic Regression model
- Converts text into TF-IDF features
- Displays the predicted category and model confidence score
- Includes a responsive interface with light and dark themes

## Built With

- Python
- Flask
- scikit-learn
- TF-IDF
- HTML and CSS

## Getting Started

### Requirements

Install Python and open a terminal in the project folder.

### Installation

```bash
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## How It Works

1. Enter text in the form and submit it.
2. The app cleans the text and converts it into TF-IDF features.
3. The trained model predicts a category.
4. The app displays the prediction and its confidence score.

The confidence score reflects the model’s output. It is not a clinical measure or diagnosis.

## Project Structure

```text
MindSense-AI/
├── app.py
├── logistic_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── static/
│   └── style.css
└── templates/
    └── index.html
```

## Note

The model and vectorizer files must be present in the project folder for the app to run.
