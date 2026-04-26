import numpy as np
import pickle
import os
import re
import nltk
import requests
from bs4 import BeautifulSoup
from django.shortcuts import render
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from .models import NewsHistory

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# --- INITIALIZATION ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'detector', 'model.pkl')
vect_path  = os.path.join(BASE_DIR, 'detector', 'tfidfvect.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    model = None
    vectorizer = None

nltk.download('stopwords', quiet=True)
ps        = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ─────────────────────────────────────────────
# LISTS
# ─────────────────────────────────────────────

FACT_CHECK_LIST = {
    "earth is round":                   "Real News (Universal Truth)",
    "water boils at 100":               "Real News (Scientific Fact)",
    "sun rises in the east":            "Real News (Universal Truth)",
    "python is a programming language": "Real News (Technical Fact)",
}

# Sensationalist / misinformation keywords → strong fake signal
RED_FLAGS = [
    'leaked', 'mass panic', 'shocking truth', 'conspiracy', 'hoax',
    'giant penguins', 'hollow earth', 'blue sun', 'naturalnews',
    'share before this gets deleted', 'they dont want you to know',
    'wake up sheeple', 'deep state', 'plandemic', 'false flag',
    'new world order', 'illuminati', 'chemtrails',
    'microchip vaccine', 'crisis actor', 'lizard people',
]

# Domains confirmed unreliable
UNRELIABLE_DOMAINS = [
    'naturalnews.com', 'infowars.com', 'beforeitsnews.com',
    'yournewswire.com', 'worldnewsdailyreport.com',
    'realnewsrightnow.com', 'empirenews.net',
]

# Domains confirmed reliable
RELIABLE_DOMAINS = [
    # Wire / international
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk',
    'aljazeera.com', 'bloomberg.com', 'economist.com', 'ft.com',
    'theguardian.com', 'nytimes.com', 'washingtonpost.com',
    'npr.org', 'pbs.org', 'dw.com', 'france24.com',
    'abc.net.au', 'cbc.ca', 'politico.com', 'theatlantic.com',
    # India
    'timesofindia.com', 'thehindu.com', 'ndtv.com',
    'hindustantimes.com', 'theprint.in', 'scroll.in',
    'wionews.com', 'indiatoday.in', 'indianexpress.com',
    'livemint.com', 'businessstandard.com', 'deccanherald.com',
    'tribuneindia.com', 'newindianexpress.com',
]

# ─────────────────────────────────────────────
# CREDIBILITY SCORING
# Each signal adds points. Higher score = more journalistic.
# ─────────────────────────────────────────────

CREDIBILITY_SIGNALS = {
    # News agency bylines — strongest signal
    'reuters':       4, 'associated press': 4, '– ians': 4,
    '- ians': 4, '–ians': 4, '-ians': 4, 'ians':  3,
    '– pti':  4, '- pti':  4, 'pti':    3,
    '– ani':  4, '- ani':  4, 'ani':    3,
    '– afp':  4, '- afp':  4, 'afp':    3,

    # Attribution phrases
    'said on':          3, 'said the':        3, 'he said':      3,
    'she said':         3, 'they said':        3, 'said that':    3,
    'told reporters':   3, 'told the press':   3,
    'in a statement':   3, 'said in a statement': 3,
    'confirmed that':   3, 'announced that':   3,
    'according to':     3, 'sources said':     2,
    'officials said':   2, 'government said':  2,
    'spokesperson said':3, 'minister said':    3,
    'minister stated':  3, 'minister highlighted': 3,
    'minister pointed': 3, 'expressed confidence': 2,

    # Formal institutional references
    'prime minister':   2, 'president':        2,
    'union minister':   3, 'chief minister':   3,
    'chief guest':      2, 'embassy':          3,
    'bilateral':        3, 'diplomatic':       3,
    'parliament':       2, 'ministry of':      3,
    'supreme court':    2, 'high court':       2,
    'national day':     2, 'state visit':      3,
    'joint statement':  3, 'signed a deal':    3,
    'signed deals':     3, 'signed an agreement': 3,
    'memorandum':       3, 'mou':              3,

    # Research / data attribution
    'study found':      2, 'research shows':   2,
    'data shows':       2, 'survey found':     2,
    'report said':      2, 'report shows':     2,

    # Geographic / formal markers
    'new delhi':        1, 'washington dc':    1,
    'united nations':   2, 'european union':   2,
    'world bank':       2, 'imf':              2,
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def clean_text(text):
    """Must be identical to clean_text in train.py"""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(w) for w in text if w not in stop_words]
    return ' '.join(text)

def get_top_keywords(text, vectorizer, n=3):
    tfidf_matrix  = vectorizer.transform([clean_text(text)])
    feature_names = vectorizer.get_feature_names_out()
    scores        = tfidf_matrix.toarray()[0]
    top_indices   = scores.argsort()[-n:][::-1]
    return [feature_names[i] for i in top_indices if scores[i] > 0]

def check_domain_reputation(url):
    url_lower = url.lower()
    for d in UNRELIABLE_DOMAINS:
        if d in url_lower:
            return 'unreliable'
    for d in RELIABLE_DOMAINS:
        if d in url_lower:
            return 'reliable'
    return None

def is_english(text):
    if not LANGDETECT_AVAILABLE:
        return True
    try:
        return langdetect.detect(text) == 'en'
    except Exception:
        return True

def get_credibility_score(text_lower):
    """
    Returns (score, matched_signals)
    score >= 6  → strong journalism
    score 3-5   → likely journalism
    score 0-2   → insufficient signals
    """
    score   = 0
    matched = []
    for signal, weight in CREDIBILITY_SIGNALS.items():
        if signal in text_lower:
            score += weight
            matched.append(signal)
    return score, matched

def extract_text_from_url(url):
    """Returns (text, error_message)"""
    try:
        headers = {
            'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None, f"Could not access the URL (HTTP {response.status_code}). The site may be blocking scrapers."

        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            return None, f"URL does not point to a webpage (Content-Type: {content_type}). Please paste the article text directly."

        soup    = BeautifulSoup(response.text, 'html.parser')
        article = soup.find('article')
        text    = article.get_text(separator=' ', strip=True) if article else \
                  " ".join(p.get_text() for p in soup.find_all('p'))

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


# ─────────────────────────────────────────────
# MAIN VIEW
# ─────────────────────────────────────────────

def predict_news(request):
    history = NewsHistory.objects.all().order_by('-created_at')[:6]

    if request.method == 'POST':
        input_data = request.POST.get('message', '').strip()

        if not input_data:
            return render(request, 'index.html', {'error': "Please enter some text or a URL.", 'history': history})
        if len(input_data) < 20:
            return render(request, 'index.html', {'error': "Input too short! Please enter at least a full sentence or a valid URL.", 'history': history})
        if model is None or vectorizer is None:
            return render(request, 'index.html', {'error': "Model not found. Please run train.py and place model.pkl + tfidfvect.pkl in the detector/ folder.", 'history': history})

        text       = ""
        output     = "Unknown"
        confidence = 0.0
        reasoning  = "No specific patterns detected."
        domain_rep = None

        # ── 1. URL or plain text ──────────────────────────
        if input_data.startswith(('http://', 'https://')):
            domain_rep = check_domain_reputation(input_data)
            text, err  = extract_text_from_url(input_data)
            if err:
                return render(request, 'index.html', {'error': err, 'history': history})
        else:
            text = input_data

        if not is_english(text):
            return render(request, 'index.html', {'error': "This model was trained on English news only. Please enter English text.", 'history': history})

        if len(re.findall(r'[a-zA-Z]+', text)) < 5:
            return render(request, 'index.html', {'error': "Not enough readable words found. Please enter a proper news article or headline.", 'history': history})

        clean_input = text.lower()

        # ── 2. Universal Truths ───────────────────────────
        for fact_key, fact_val in FACT_CHECK_LIST.items():
            if fact_key in clean_input:
                output     = fact_val
                confidence = 100.0
                reasoning  = f"Matches a verified factual statement ('{fact_key}') in the internal knowledge base."
                NewsHistory.objects.create(news_text=text[:200], prediction=output, confidence=confidence, reasoning=reasoning)
                return render(request, 'index.html', {
                    'output': output, 'confidence': confidence, 'reasoning': reasoning,
                    'history': NewsHistory.objects.all().order_by('-created_at')[:6],
                })

        # ── 3. Unreliable domain ──────────────────────────
        if domain_rep == 'unreliable':
            output     = "Fake News"
            confidence = 95.0
            reasoning  = "This URL is from a domain with a known history of publishing misinformation."
            NewsHistory.objects.create(news_text=text[:200], prediction=output, confidence=confidence, reasoning=reasoning)
            return render(request, 'index.html', {
                'output': output, 'confidence': confidence, 'reasoning': reasoning,
                'history': NewsHistory.objects.all().order_by('-created_at')[:6],
            })

        # ── 4. Red Flag check ─────────────────────────────
        found_flags = [w for w in RED_FLAGS if w in clean_input]
        if found_flags:
            output     = "Fake News"
            confidence = 92.0
            reasoning  = f"Flagged due to sensationalist keywords: {', '.join(found_flags)}."
            NewsHistory.objects.create(news_text=text[:200], prediction=output, confidence=confidence, reasoning=reasoning)
            return render(request, 'index.html', {
                'output': output, 'confidence': confidence, 'reasoning': reasoning,
                'history': NewsHistory.objects.all().order_by('-created_at')[:6],
            })

        # ── 5. ML Prediction ──────────────────────────────
        cleaned = clean_text(text)
        if not cleaned.strip():
            return render(request, 'index.html', {'error': "Text could not be processed. Please enter proper English news text.", 'history': history})

        vect       = vectorizer.transform([cleaned])
        val        = model.predict(vect)[0]
        proba      = model.predict_proba(vect)[0]
        prob_fake  = round(proba[0] * 100, 2)
        prob_real  = round(proba[1] * 100, 2)
        confidence = round(max(proba) * 100, 2)
        ml_real    = (val == 1)

        important_words = get_top_keywords(text, vectorizer)
        word_list       = ", ".join(important_words) if important_words else "general patterns"

        # ── 6. Credibility scoring ────────────────────────
        cred_score, cred_matched = get_credibility_score(clean_input)

        # ── 7. Final Decision Logic ───────────────────────
        #
        # Priority (highest to lowest):
        #   A. Reliable domain                    → Real News
        #   B. Credibility score ≥ 6 (strong)     → Real News  (regardless of ML)
        #   C. Credibility score 3-5 (likely)     → Real News (Likely)  if ML uncertain
        #   D. ML says Real + any cred signals    → Real News
        #   E. ML says Real, no cred signals      → Real News
        #   F. ML says Fake + cred score < 3      → Fake News
        # ─────────────────────────────────────────────────

        if domain_rep == 'reliable':
            output     = "Real News"
            confidence = max(confidence, 80.0)
            reasoning  = (
                f"From a verified reliable source. "
                f"Model scores — Fake: {prob_fake}%, Real: {prob_real}%. "
                f"Domain trust applied. Key terms: {word_list}."
            )

        elif cred_score >= 6:
            # Strong journalistic content — override ML regardless
            output     = "Real News"
            confidence = max(confidence if ml_real else (100 - confidence), 75.0)
            reasoning  = (
                f"Strong journalistic structure detected (credibility score: {cred_score}). "
                f"Matched signals: {', '.join(cred_matched[:5])}. "
                f"Key terms: {word_list}."
            )

        elif cred_score >= 3:
            # Moderate journalism signals
            if ml_real:
                output     = "Real News"
                confidence = max(confidence, 72.0)
                reasoning  = (
                    f"ML model and journalistic signals agree (credibility score: {cred_score}). "
                    f"Fake: {prob_fake}%, Real: {prob_real}%. Key terms: {word_list}."
                )
            else:
                # ML says fake but journalism signals present — call it uncertain
                output     = "Real News (Likely)"
                confidence = max(100 - confidence, 65.0)
                reasoning  = (
                    f"Journalistic writing style detected (credibility score: {cred_score}) "
                    f"despite model leaning Fake ({prob_fake}%). "
                    f"Matched signals: {', '.join(cred_matched[:4])}. "
                    f"Key terms: {word_list}."
                )

        elif ml_real:
            # ML says real, low/no credibility signals
            output    = "Real News"
            reasoning = (
                f"Linguistic analysis indicates credible journalism style. "
                f"Fake: {prob_fake}%, Real: {prob_real}%. Key terms: {word_list}."
            )

        else:
            # ML says fake, no credibility signals to override
            output    = "Fake News"
            reasoning = (
                f"Linguistic patterns consistent with misinformation. "
                f"Fake: {prob_fake}%, Real: {prob_real}%. Key terms: {word_list}."
            )

        # ── 8. Save & Render ──────────────────────────────
        NewsHistory.objects.create(
            news_text=text[:200],
            prediction=output,
            confidence=confidence,
            reasoning=reasoning,
        )

        return render(request, 'index.html', {
            'output':     output,
            'confidence': confidence,
            'reasoning':  reasoning,
            'history':    NewsHistory.objects.all().order_by('-created_at')[:6],
        })

    return render(request, 'index.html', {'history': history})