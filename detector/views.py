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

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vect_path, 'rb') as f:
    vectorizer = pickle.load(f)

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
    # Transform the single input
    tfidf_matrix = vectorizer.transform([clean_text(text)])
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the scores for this specific text
    scores = tfidf_matrix.toarray()[0]
    
    # Find the indices of the highest scoring words
    top_indices = scores.argsort()[-n:][::-1]
    
    # Return the actual words
    return [feature_names[i] for i in top_indices if scores[i] > 0]

# --- MAIN PREDICTION VIEW ---
def predict_news(request):
    history = NewsHistory.objects.all().order_by('-created_at')[:5]

    if request.method == 'POST':
        input_data = request.POST.get('message', '').strip()
        
        if not input_data:
            return render(request, 'index.html', {'error': "Please enter some text or a link!", 'history': history})

        # --- NEW: URL SCRAPER LOGIC ---
        if input_data.startswith(('http://', 'https://')):
            try:
                # Adding headers to mimic a browser so sites don't block us
                # Replace your current headers with this detailed one
                headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/'
            }
                response = requests.get(input_data, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text from all paragraph tags
                paragraphs = soup.find_all('p')
                text = " ".join([p.get_text() for p in paragraphs])
                
                if len(text) < 100:
                    return render(request, 'index.html', {'error': "Could not extract enough text from the link. Try a different site.", 'history': history})
            except Exception as e:
                return render(request, 'index.html', {'error': f"Error fetching link: {e}", 'history': history})
        else:
            text = input_data

        # 1. Step One: Check for Universal Truths
        clean_input = text.lower()
        for fact in FACT_CHECK_LIST:
            if fact in clean_input:
                output = FACT_CHECK_LIST[fact]
                confidence = 100.0
                NewsHistory.objects.create(news_text=text[:200], prediction=output, confidence=confidence)
                return render(request, 'index.html', {
                    'output': output, 'confidence': confidence, 
                    'original_text': text[:500], 'history': history
                })

        # 2. Step Two: ML Prediction
        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])
        val = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]
        confidence = round(max(proba) * 100, 2)

        # DEBUG PRINTS
        print(f"\nRAW MODEL VAL: {val} | PROBABILITY: {proba}\n")

        # ... inside your predict_news function, after the ML Prediction step ...

        # Get the specific words the model focused on
        important_words = get_top_keywords(text, vectorizer)
        word_list = ", ".join(important_words)

        if is_suspicious:
            output = "Fake News"
            # Find which specific red flag was caught
            found_flags = [word for word in red_flags if word in clean_input]
            reasoning = f"This content was flagged as Fake because it contains sensationalist phrases like '{', '.join(found_flags)}'."
        
        elif val == 1:
            output = "Real News"
            reasoning = f"The model categorized this as Real because of the professional use of terms like '{word_list}', which are frequently found in credible news reporting."
        
        else:
            output = "Fake News"
            reasoning = f"The model detected patterns common in misinformation, specifically focusing on the context of words like '{word_list}'."

        # 3. Step Three: Logic Overlays (Clickbait & Absurdity)
        red_flags = [
            'leaked', 'secret', 'exposed', 'mass panic', 'breaking',
            'shocking truth', 'conspiracy', 'hoax', 'giant penguins', 
            'green eyes', 'hollow earth', 'blue sun',
            'surprised to lose', 'miracle cure', 'aliens', 'unexplained' # Added satire/hoax words
        ]
        
        is_suspicious = any(word in clean_input for word in red_flags)

        if is_suspicious:
            output = "Fake News"
            confidence = max(confidence, 92.0) 
        elif confidence < 80:
            output = "Fake News (Unverified)"
        else:
            output = "Real News" if val == 1 else "Fake News"
        
        # 4. Save Prediction to History
        NewsHistory.objects.create(news_text=text[:200], prediction=output, confidence=confidence)
        history = NewsHistory.objects.all().order_by('-created_at')[:5]

        return render(request, 'index.html', {
            'output': output,
            'confidence': confidence,
            'original_text': text[:500] + ("..." if len(text) > 500 else ""),
            'history': history
        })

    return render(request, 'index.html', {'history': history})