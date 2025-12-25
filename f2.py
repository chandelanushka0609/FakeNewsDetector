import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# -------------------------
# Load dataset.csv
# -------------------------
df = pd.read_csv("dataset.csv")

# -------------------------
# Preprocessing
# -------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean"] = df["text"].apply(clean_text)

# -------------------------
# Vectorize
# -------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean"]).toarray()
y = df["label"].map({"real": 1, "fake": 0}).values

# -------------------------
# Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train
# -------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# -------------------------
# Save
# -------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("model.pkl and vectorizer.pkl saved!")
