📰 Fake News Detection System
Machine Learning + Streamlit Web Application

An interactive Fake News Detection System built using Machine Learning and a modern Streamlit-based web interface.
The application analyzes news content and predicts whether it is Real or Fake, along with a credibility score displayed through an animated visualization.

The system supports:

Manual news text input

Live news article analysis via URL

Bulk classification using CSV files

✨ Key Features
🔍 Real-Time Fake News Detection

Paste any news text or

Enter a news article URL to automatically extract and analyze content

🤖 Machine Learning Model

Logistic Regression classifier

TF-IDF vectorization with 5000 features

Trained on a labeled fake/real news dataset

🎨 Interactive & Animated UI

Aurora Neon themed interface

Animated credibility gauge

Gradient prediction badges

Smooth transitions for better user experience

📂 Batch Processing (CSV Upload)

Upload CSV files with multiple news articles

Classify hundreds of entries in one click

Download results instantly

🌐 URL-Based News Analysis

Automatically extracts article text using:

newspaper3k

readability-lxml

Fallback scraper using BeautifulSoup

🛠️ Tech Stack
Component	Technology
Model	Logistic Regression
Feature Extraction	TF-IDF Vectorizer
Frontend	Streamlit
Programming Language	Python
Dataset	Custom combined fake & real news data
📁 Project Structure
Fake-News-Detector/
│
├── app.py              # Main Streamlit application
├── f2.py               # Model training script
├── model.pkl           # Trained ML model
├── vectorizer.pkl      # Saved TF-IDF vectorizer
├── dataset.csv         # Training dataset
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

🚀 Installation & Setup
1️⃣ Clone the repository
git clone <your-repository-link>
cd Fake-News-Detector

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
streamlit run app.py


The web app will open automatically in your browser.

⚙️ How the System Works
🧹 Text Preprocessing

Converts text to lowercase

Removes punctuation

Eliminates stopwords

Applies lemmatization

Converts text into numerical form using TF-IDF

📊 Prediction Output

For each input, the model provides:

Fake News (0) or Real News (1)

Confidence probability

Visual credibility gauge (%)

📑 Batch Analysis (CSV Upload)

Prepare a CSV file with a text column:

text
"Sample news article text..."
"Another news article text..."


Upload the file in the app → get predictions instantly → download the result CSV.

🌍 Real-Time URL Checker

Paste any valid news article URL

System extracts the article content

Runs the ML pipeline

Displays prediction + credibility score

👩‍🏫 For Evaluation / Demo Use

Install required libraries

Run the Streamlit app

Test using:

Manual text input

URL-based analysis

CSV bulk upload

No additional configuration is required.

🔮 Future Enhancements

Transformer-based models (BERT, DistilBERT)

Multi-language news detection

Browser extension integration

Cloud-based API deployment

👤 Developer

Your Name

Fake News Detection System — 2025
Built using Machine Learning and Streamlit.# FakeNewsDetector
