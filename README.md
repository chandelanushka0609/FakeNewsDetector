# Fake News Detection System

A simple Fake News Detection web application built using Machine Learning and Streamlit.
The system predicts whether a news article is Fake or Real and displays a credibility score.

## 1. Project Overview
This project uses a Logistic Regression model trained on labeled news data.
Users can analyze news by pasting text, entering a URL, or uploading a CSV file.

## 2. Features
1. Detects fake or real news from pasted text
2. Supports real-time news checking using article URLs
3. Allows bulk prediction using CSV file upload
4. Displays prediction confidence as a credibility score
5. Simple and interactive Streamlit web interface

## 3. Tech Stack
1. Python
2. Logistic Regression
3. TF-IDF Vectorizer
4. Streamlit

## 4. Project Structure
1. app.py – Streamlit web application
2. f2.py – Model training script
3. model.pkl – Trained ML model
4. vectorizer.pkl – Saved TF-IDF vectorizer
5. dataset.csv – Training dataset
6. requirements.txt – Required libraries

## 5. How to Run the Project
1. Clone the repository
2. Install dependencies using: pip install -r requirements.txt
3. Run the app using: streamlit run app.py

## 6. Output
1. Fake News or Real News prediction
2. Confidence score shown as percentage

## 7. Future Improvements
1. Use advanced models like BERT
2. Add multi-language support
3. Deploy as a cloud-based API

## 8. Author
Anushka Chandel

Fake News Detection Project – 2025


