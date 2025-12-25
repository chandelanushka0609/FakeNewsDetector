# Fake News Detection System (Machine Learning + Streamlit)
An interactive Fake News Classification System built using Logistic Regression, TF-IDF Vectorization, and a modern Streamlit Web App (Aurora Neon UI). The system can analyze:

Manually pasted news text

Real-time online news articles (via URL extraction)

Bulk CSV files with multiple news entries

It produces a credibility score, a fake/real prediction, and includes a beautiful animated gauge visualization.

Features
Real-time Fake News Detection:

Paste text or enter a URL — the system extracts and analyzes the article.

Machine Learning Model (Logistic Regression):

Trained on labeled fake/real news datasets using TF-IDF (5000 features).

Animated UI Components:

Aurora Neon Theme

Animated credibility gauge

Gradient badges

Smooth transitions

Batch Processing:

Upload a CSV containing a text column to classify hundreds of articles at once.

URL Content Extraction:

Supports:

newspaper3k

readability-lxml

Fallback: BeautifulSoup scraper

Tech Stack
(Component -> Technology)

Model -> Logistic Regression

Feature Extraction -> TfidfVectorizer (5000 features)

Frontend -> Streamlit

Dataset -> Custom combined news dataset

Language -> Python

Project Structure
Fake-News-Detector/

── app.py # Main Streamlit application

── f2.py # Model training script

── model.pkl # Trained ML model

── vectorizer.pkl # Saved TF-IDF vectorizer

── dataset.csv # Training dataset

── requirements.txt # Required libraries

── README.md # Documentation

Installation & Setup
1️. Clone the repository:

git clone <your-repo-link>

cd Fake-News-Detector
2️. Install all dependencies:

pip install -r requirements.txt
3️. Run the web app:

streamlit run app.py
The application will open in your browser automatically.

How It Works
Preprocessing:

Lowercase conversion

Removing punctuation

Stopword removal

Lemmatization

TF-IDF vectorization

Model Prediction:

For each input, the model outputs:

Fake News (0)

Real News (1)

Probability score → displayed via animated gauge

UI Output:

Prediction category

Credibility score (%)

Animation badge

Batch Analysis (CSV Upload)
Prepare a CSV file with at least:

text

"Some news article..."

"Another news article..."

Then upload in the web app → results appear instantly + downloadable CSV.

Real-Time URL Checker
Paste any article URL →

System extracts content → runs the same ML pipeline → returns prediction + score.

For Teachers / Evaluators
Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app.py

Test using:

Manually typed news

URL extraction

Bulk CSV upload

No external configuration is required.

Future Improvements
Transformer-based models (BERT, DistilBERT)

Multi-language news detection

Browser plugin extension

API deployment on cloud

Developer
Ayush

(Fake News Detection Project — 2025)

(Built with love and AI assistance.)
