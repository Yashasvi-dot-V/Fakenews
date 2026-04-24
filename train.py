import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ✅ FIX: Apply the SAME clean_text during training that views.py uses at prediction time
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

print("--- 1. Loading WELFake Dataset ---")
try:
    df = pd.read_csv('WELFake_Dataset.csv')
    df = df.dropna()
    print(f"Loaded {len(df)} rows")
    print("Label distribution:")
    print(df['label'].value_counts())
    # ✅ FIX: Correct label mapping for WELFake dataset
    # WELFake: 1 = FAKE, 0 = REAL (NOT the other way around!)
    print("\nVerification — label 0 sample title:", df[df['label']==0].iloc[0]['title'][:80])
    print("Verification — label 1 sample title:", df[df['label']==1].iloc[0]['title'][:80])
except FileNotFoundError:
    print("Error: WELFake_Dataset.csv not found!")
    exit()

print("\n--- 2. Cleaning & Combining Text ---")
# ✅ FIX: Apply clean_text HERE during training so TF-IDF vocabulary matches prediction input
df['total_content'] = (df['title'] + " " + df['text']).apply(clean_text)

print("--- 3. Vectorizing ---")
tfidf = TfidfVectorizer(max_df=0.7, ngram_range=(1, 2), max_features=10000)
X = tfidf.fit_transform(df['total_content'])
y = df['label']  # ✅ WELFake: 1 = FAKE, 0 = REAL

print("--- 4. Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

print("--- 5. Training Random Forest ---")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print("--- 6. Accuracy Check ---")
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"Final Accuracy: {accuracy * 100:.2f}%")

print("--- 7. Saving Model Files ---")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidfvect.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\n✅ Done! model.pkl and tfidfvect.pkl saved.")
print("Label reminder: model predicts 1 = FAKE, 0 = REAL")