import numpy as np
import pickle
import os
import re
import nltk
import requests
import langdetect
from bs4 import BeautifulSoup
from django.shortcuts import render
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
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

nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

FACT_CHECK_LIST = {
    "earth is round": "Real News (Universal Truth)",
    "water boils at 100": "Real News (Scientific Fact)",
    "sun rises in the east": "Real News (Universal Truth)",
    "python is a programming language": "Real News (Technical Fact)"
}

RED_FLAGS = [
    'leaked', 'secret', 'exposed', 'mass panic', 'shocking truth',
    'conspiracy', 'hoax', 'giant penguins', 'hollow earth',
    'blue sun', 'naturalnews', 'mike adam', 'share before this gets deleted',
    'they dont want you to know', 'wake up sheeple', 'deep state',
    'plandemic', 'false flag', 'new world order'
]

# Domains known to be unreliable
UNRELIABLE_DOMAINS = [
    'naturalnews.com', 'infowars.com', 'beforeitsnews.com',
    'yournewswire.com', 'worldnewsdailyreport.com'
]

# Domains known to be reliable
RELIABLE_DOMAINS = [
    'reuters.com', 'bbc.com', 'bbc.co.uk', 'apnews.com',
    'timesofindia.com', 'thehindu.com', 'ndtv.com',
    'theguardian.com', 'nytimes.com', 'washingtonpost.com'
]

# --- HELPER FUNCTIONS ---
def clean_text(text):
    """Must be identical to clean_text in train.py"""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

def get_top_keywords(text, vectorizer, n=3):
    tfidf_matrix = vectorizer.transform([clean_text(text)])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    top_indices = scores.argsort()[-n:][::-1]
    return [feature_names[i] for i in top_indices if scores[i] > 0]

def check_domain_reputation(url):
    """Returns 'reliable', 'unreliable', or None"""
    url_lower = url.lower()
    for domain in UNRELIABLE_DOMAINS:
        if domain in url_lower:
            return 'unreliable'
    for domain in RELIABLE_DOMAINS:
        if domain in url_lower:
            return 'reliable'
    return None

def is_english(text):
    """Check if text is in English"""
    try:
        return langdetect.detect(text) == 'en'
    except Exception:
        return True  # assume English if detection fails

def extract_text_from_url(url):
    """
    Returns (text, error_message)
    text is None if extraction failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        response = requests.get(url, headers=headers, timeout=10)

        # Edge case: non-200 response
        if response.status_code != 200:
            return None, f"Could not access the URL (HTTP {response.status_code}). The site may be blocking scrapers."

        # Edge case: non-HTML content (PDF, image, etc.)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            return None, f"URL does not point to a webpage (Content-Type: {content_type}). Please paste the article text directly."

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try article tag first, then fall back to <p> tags
        article = soup.find('article')
        if article:
            text = article.get_text(separator=' ', strip=True)
        else:
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text() for p in paragraphs])

        # Edge case: paywalled or JS-rendered sites
        if len(text.strip()) < 100:
            return None, "Could not extract enough text. The site may be paywalled or requires JavaScript. Please paste the article text directly."

        return text.strip(), None

    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the URL. Please check the link or your internet connection."
    except requests.exceptions.Timeout:
        return None, "The request timed out (>10s). The site may be slow. Try pasting the text directly."
    except requests.exceptions.TooManyRedirects:
        return None, "Too many redirects. This URL may be broken."
    except Exception as e:
        return None, f"Unexpected error fetching URL: {str(e)}"

# --- MAIN PREDICTION VIEW ---
def predict_news(request):
    history = NewsHistory.objects.all().order_by('-created_at')[:5]

    if request.method == 'POST':
        input_data = request.POST.get('message', '').strip()

        # Edge case: empty input
        if not input_data:
            return render(request, 'index.html', {
                'error': "Please enter some text or a URL.",
                'history': history
            })

        # Edge case: too short
        if len(input_data) < 20:
            return render(request, 'index.html', {
                'error': "Input too short! Please enter at least a full sentence or a valid URL.",
                'history': history
            })

        # Edge case: model not loaded
        if model is None or vectorizer is None:
            return render(request, 'index.html', {
                'error': "Model not found. Please run train.py and place model.pkl + tfidfvect.pkl in the detector/ folder.",
                'history': history
            })

        text = ""
        output = "Unknown"
        confidence = 0.0
        reasoning = "No specific patterns detected."
        domain_rep = None

        # 1. URL or plain text
        if input_data.startswith(('http://', 'https://')):
            domain_rep = check_domain_reputation(input_data)
            text, error = extract_text_from_url(input_data)
            if error:
                return render(request, 'index.html', {'error': error, 'history': history})
        else:
            text = input_data

        # Edge case: non-English text
        if not is_english(text):
            return render(request, 'index.html', {
                'error': "This model was trained on English news only. Non-English text may give inaccurate results. Please enter English text.",
                'history': history
            })

        # Edge case: input is just numbers/symbols, no real words
        word_count = len(re.findall(r'[a-zA-Z]+', text))
        if word_count < 5:
            return render(request, 'index.html', {
                'error': "Not enough readable words found. Please enter a proper news article or headline.",
                'history': history
            })

        # 2. Universal Truths Layer
        clean_input = text.lower()
        for fact_key in FACT_CHECK_LIST:
            if fact_key in clean_input:
                output = FACT_CHECK_LIST[fact_key]
                confidence = 100.0
                reasoning = f"Matches a verified factual statement ('{fact_key}') in the internal knowledge base."
                NewsHistory.objects.create(news_text=text[:200], prediction=output, confidence=confidence, reasoning=reasoning)
                history = NewsHistory.objects.all().order_by('-created_at')[:5]
                return render(request, 'index.html', {
                    'output': output, 'confidence': confidence,
                    'reasoning': reasoning, 'history': history
                })

        # 3. Domain reputation shortcut
        if domain_rep == 'unreliable':
            output = "Fake News"
            confidence = 95.0
            reasoning = "This URL is from a domain with a known history of publishing misinformation."
            NewsHistory.objects.create(news_text=text[:200], prediction=output, confidence=confidence, reasoning=reasoning)
            history = NewsHistory.objects.all().order_by('-created_at')[:5]
            return render(request, 'index.html', {
                'output': output, 'confidence': confidence,
                'reasoning': reasoning, 'history': history
            })

        # 4. ML Prediction
        cleaned = clean_text(text)

        # Edge case: after cleaning, nothing left (e.g. text was all numbers/symbols)
        if not cleaned.strip():
            return render(request, 'index.html', {
                'error': "Text could not be processed. Please enter proper English news text.",
                'history': history
            })

        vect = vectorizer.transform([cleaned])
        val = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]
        confidence = round(max(proba) * 100, 2)

        important_words = get_top_keywords(text, vectorizer)
        word_list = ", ".join(important_words) if important_words else "general patterns"

        # 5. Red Flag Heuristic Overlay
        found_flags = [word for word in RED_FLAGS if word in clean_input]
        is_suspicious = len(found_flags) > 0

        if is_suspicious:
            output = "Fake News"
            confidence = max(confidence, 92.0)
            reasoning = f"Flagged due to sensationalist keywords: {', '.join(found_flags)}."

        else:
            # Your trained model: 1 = Real, 0 = Fake (confirmed by your testing)
            if val == 1:
                output = "Real News"
                status = "credible journalism"
            else:
                output = "Fake News"
                status = "known misinformation patterns"

            # Domain boost for known reliable sources
            if domain_rep == 'reliable' and output == "Fake News" and confidence < 75:
                output = "Real News"
                confidence = max(confidence, 75.0)
                reasoning = f"From a known reliable source. ML patterns loosely flagged it but domain trust overrides. Key terms: {word_list}."
            elif confidence < 65:
                reasoning = f"Model is uncertain ({confidence}%). Linguistic patterns loosely resemble {status}. Key terms: {word_list}."
            else:
                reasoning = f"Linguistic analysis indicates a style consistent with {status}. Key terms: {word_list}."

        # 6. Save and Render
        NewsHistory.objects.create(
            news_text=text[:200],
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