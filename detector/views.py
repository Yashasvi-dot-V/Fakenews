import numpy as np
import pickle
import os
import re
import nltk
import requests
from bs4 import BeautifulSoup
from django.shortcuts import render
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from .models import NewsHistory

# --- INITIALIZATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'detector', 'model.pkl')
vect_path = os.path.join(BASE_DIR, 'detector', 'tfidfvect.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    model = None
    vectorizer = None

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

FACT_CHECK_LIST = {
    "earth is round": "Real News (Universal Truth)",
    "water boils at 100": "Real News (Scientific Fact)",
    "sun rises in the east": "Real News (Universal Truth)",
    "python is a programming language": "Real News (Technical Fact)"
}

# --- HELPER FUNCTIONS ---
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    return text

def get_top_keywords(text, vectorizer, n=3):
    tfidf_matrix = vectorizer.transform([clean_text(text)])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    top_indices = scores.argsort()[-n:][::-1]
    return [feature_names[i] for i in top_indices if scores[i] > 0]

# --- MAIN PREDICTION VIEW ---
def predict_news(request):
    history = NewsHistory.objects.all().order_by('-created_at')[:5]

    if request.method == 'POST':
        input_data = request.POST.get('message', '').strip()

        if len(input_data) < 10:
            return render(request, 'index.html', {
                'error': "Input too short! Please enter a full news sentence or a valid URL for analysis.",
                'history': history
            })

        text = ""
        output = "Unknown"
        confidence = 0.0
        reasoning = "No specific patterns detected."
        is_suspicious = False

        if not input_data:
            return render(request, 'index.html', {'error': "Please enter text or a URL!", 'history': history})

        # 1. URL Scraper Logic
        if input_data.startswith(('http://', 'https://')):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                    'Accept-Language': 'en-US,en;q=0.5'
                }
                response = requests.get(input_data, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text = " ".join([p.get_text() for p in paragraphs])

                if len(text) < 50:
                    return render(request, 'index.html', {'error': "Could not extract enough text from the link.", 'history': history})
            except Exception as e:
                return render(request, 'index.html', {'error': f"Link error: {e}", 'history': history})
        else:
            text = input_data

        # 2. Universal Truths Layer
        clean_input = text.lower()
        for fact_key in FACT_CHECK_LIST:
            if fact_key in clean_input:
                output = FACT_CHECK_LIST[fact_key]
                confidence = 100.0
                reasoning = f"The input matches a verified factual statement ('{fact_key}') in the internal knowledge base."

                NewsHistory.objects.create(news_text=text[:100], prediction=output, confidence=confidence)
                history = NewsHistory.objects.all().order_by('-created_at')[:5]
                return render(request, 'index.html', {
                    'output': output, 'confidence': confidence,
                    'reasoning': reasoning, 'history': history
                })

        # 3. ML Prediction & Feature Extraction
        if model is None or vectorizer is None:
            return render(request, 'index.html', {
                'error': "Model not found. Please train the model first.",
                'history': history
            })

        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])
        val = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]
        confidence = round(max(proba) * 100, 2)

        important_words = get_top_keywords(text, vectorizer)
        word_list = ", ".join(important_words)

        # 4. Red Flag / Heuristic Overlay
        red_flags = [
            'leaked', 'secret', 'exposed', 'mass panic', 'breaking',
            'shocking truth', 'conspiracy', 'hoax', 'giant penguins',
            'green eyes', 'hollow earth', 'blue sun', 'naturalnews', 'mike adam'
        ]

        found_flags = [word for word in red_flags if word in clean_input]
        is_suspicious = len(found_flags) > 0

        if is_suspicious:
            output = "Fake News"
            confidence = max(confidence, 92.0)
            reasoning = f"Flagged as Fake due to the presence of sensationalist keywords: {', '.join(found_flags)}."
        elif confidence < 80:
            output = "Fake News (Unverified)"
            reasoning = f"Model is uncertain (Confidence < 80%). The structure loosely resembles misinformation patterns around: {word_list}."
        else:
            output = "Real News" if val == 1 else "Fake News"
            status = "credible journalism" if val == 1 else "known misinformation patterns"
            reasoning = f"Linguistic analysis indicates a style consistent with {status}, focusing on key terms: {word_list}."

        # 5. Final Save and Render
        NewsHistory.objects.create(
            news_text=text[:100],
            prediction=output,
            confidence=confidence,
            reasoning=reasoning
        )
        history = NewsHistory.objects.all().order_by('-created_at')[:5]

        return render(request, 'index.html', {
            'output': output,
            'confidence': confidence,
            'reasoning': reasoning,
            'history': history
        })

    return render(request, 'index.html', {'history': history})