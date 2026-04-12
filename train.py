import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("--- 1. Loading WELFake Dataset ---")
try:
    df = pd.read_csv('WELFake_Dataset.csv')
    df = df.dropna()
    # Combining Title and Text gives the model more 'features' to look at
    df['total_content'] = df['title'] + " " + df['text']
except FileNotFoundError:
    print("Error: WELFake.csv not found in this folder!")
    exit()

print("--- 2. Vectorizing (Using Bi-grams for higher accuracy) ---")
# ngram_range=(1,2) helps the model understand context/facts
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2), max_features=10000)
X = tfidf.fit_transform(df['total_content'])
y = df['label'] # 1 for Real, 0 for Fake

print("--- 3. Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- 4. Training Random Forest (Using your Ryzen cores) ---")
# n_jobs=-1 makes training much faster on your laptop
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print("--- 5. Accuracy Check ---")
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"Final Accuracy: {accuracy * 100:.2f}%")

print("--- 6. Saving New Model Files ---")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidfvect.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Success! New .pkl files created.")