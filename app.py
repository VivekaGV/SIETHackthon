from flask import Flask, render_template,request,redirect,session,url_for
import os
import joblib
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import csv
from openpyxl import load_workbook
from textblob import TextBlob
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from model import recommend_top_drugs

nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('wordnet')

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

app = Flask(__name__)

app.secret_key=os.urandom(24)
# Model saved with Keras model.save()
MODEL_PATH = r"C:\Users\Vivek  G V\Downloads\DRUG_MAIN_FINAL\DRUG_MAIN_FINAL\DRUG_MAIN_FINAL\condition_classification_model.pkl"

TOKENIZER_PATH = r"C:\Users\Vivek  G V\Downloads\DRUG_MAIN_FINAL\DRUG_MAIN_FINAL\DRUG_MAIN_FINAL\condition_classification_model.pkl"

DATA_PATH = r"C:\Users\Vivek  G V\Downloads\DRUG_MAIN_FINAL\DRUG_MAIN_FINAL\DRUG_MAIN_FINAL\output.csv"

# loading vectorizer
vectorizer = joblib.load(TOKENIZER_PATH)
# loading model
# model = joblib.load(MODEL_PATH)
df = pd.read_csv('data/output.csv')

#getting stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        with open('users.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == username:
                    # If username already exists, redirect to login page
                    return redirect(url_for('login'))
        
        # If username does not exist, store user details in CSV file
        with open('users.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([username, email, password])
        
        # After successful signup, redirect to login page
        return redirect(url_for('login'))
    
    # Render signup form template
    return render_template('signup.html')


# Route for login page
@app.route('/login')
def login():
    # Render login form template
    return render_template('login.html')

@app.route('/')
def sign_up():
	return render_template('signup.html')

@app.route('/index')
def home():
	return render_template('home.html')

@app.route("/logout")
def logout():
	session.clear()
	return render_template('signup.html')

@app.route('/drugRecommend')
def index():
	return render_template('DrugRecommend.html')

@app.route('/security')
def check_security():
    # Redirect logic for security check page
    return render_template('safety.html')

@app.route('/review')
def review_drug():
    # Redirect logic for drug review page
    return render_template('review.html')

@app.route('/checkReview')
def check_review():
    # Redirect logic for checking drug reviews page
    return render_template('checkReview.html')


@app.route('/login_validation', methods=['POST'])
def login_validation():
    # Get username and password from the form
    username = request.form.get('username')
    password = request.form.get('password')

    # Check if username and password match with records in users.csv
    with open('users.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] == username and row[2] == password:
                # If username and password match, redirect to home.html
                return redirect(url_for('home'))
    
    # If username and password don't match, display error message
    error_message = "Incorrect username or password. Please try again."
    return render_template('login.html', error=error_message)

@app.route('/predict', methods=["GET", "POST"])

def predict():
    if request.method == 'POST':
        raw_text = request.form['rawtext']
        condition = detect_condition(raw_text)
        top_three_drugs = recommend_top_drugs(df, condition)
        return render_template('predict.html', rawtext=raw_text, condition=condition, top_three_drugs=top_three_drugs)

def detect_condition(raw_text):
    # Define a list of common medical conditions
    
    # Convert input text to lowercase for case-insensitive matching
    raw_text = raw_text.lower()

    conditions = df['condition'].str.lower().unique()
    
    # Initialize detected condition to None
    detected_condition = None
    
    # Loop through the list of conditions and check if any of them appear in the input text
    for condition in conditions:
        if re.search(r'\b{}\b'.format(condition), raw_text):
            detected_condition = condition
            break
    
    return detected_condition


# @app.route('/get_top_drugs/<condition>/<rawtext>')
# def get_top_drugs(condition, rawtext):
#     df = pd.read_csv(DATA_PATH)
#     top_drugs_data = top_drugs_extractor_with_percentage(condition, df)

#     return render_template('predict.html', rawtext=rawtext, result=condition, top_drugs_data=top_drugs_data)

# def cleanText(raw_review):
#     print("Received raw_review:", raw_review)
#     # 1. Delete HTML 
#     review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
#     print("After HTML removal:", review_text)
#     # 2. Make a space
#     letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
#     print("After removing non-alphabetic characters:", letters_only)
#     # 3. lower letters
#     words = letters_only.lower().split()
#     print("After converting to lowercase and splitting:", words)
#     # 5. Stopwords 
#     meaningful_words = [w for w in words if not w in stop]
#     print("After removing stopwords:", meaningful_words)
#     # 6. lemmitization
#     lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
#     print("After lemmatization:", lemmitize_words)
#     # 7. space join words
#     result = ' '.join(lemmitize_words)
#     print("Final result:", result)
#     return result



# def top_drugs_extractor_with_percentage(condition, df):
#     df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
#     top_drugs_data = []
#     for drug in df_top[df_top['condition'] == condition]['drugName'].head(3).tolist():
#         total_reviews = len(df[df['drugName'] == drug])
#         positive_reviews = len(df[(df['drugName'] == drug) & (df['sentiment'] == 'positive')])
#         if total_reviews > 0:
#             percentage = (positive_reviews / total_reviews) * 100
#         else:
#             percentage = 0
#         top_drugs_data.append({'drug': drug, 'percentage': percentage})
#     return top_drugs_data


@app.route('/add_review', methods=['POST'])
def add_review():
    if request.method == 'POST':
        unique_id = request.form['unique_id']
        drug_name = request.form['drug_name']
        condition = request.form['condition']
        review = request.form['review']
        rating = request.form['rating']
        date = request.form['date']
        useful_count = request.form['useful_count']

        # Perform sentiment analysis on the review
        blob = TextBlob(review)
        sentiment_polarity = blob.sentiment.polarity  # Get the sentiment polarity

        # Categorize sentiment
        if sentiment_polarity > 0:
            sentiment = 'Positive'
        elif sentiment_polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        # Append the entered data along with the sentiment to the existing CSV file
        with open('data/output.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([unique_id, drug_name, condition, review, rating, date, useful_count, sentiment])

        return redirect('/success')

@app.route('/success')
def success():
    message = "Review added successfully!"
    return f"<script>alert('{message}'); window.location = '/review';</script>"

@app.route('/check_safety', methods=['POST'])
def check_safety():
    if request.method == 'POST':
        drug_name = request.form['drug_name']

        # Read the dataset
        df = pd.read_csv('data/output.csv')

        # Search for the drug name in the dataset
        drug_entry = df[df['drugName'] == drug_name]

        if not drug_entry.empty:
            # Extract safety information from the drug entry
            condition = drug_entry.iloc[0]['condition']
            reviews = drug_entry['review']

            # Analyze the reviews to extract safety information
            side_effects = []
            benefits = []
            precautions = []
            warnings = []

            for review in reviews:
                blob = TextBlob(review)
                for sentence in blob.sentences:
                    # Check for patterns indicating side effects, benefits, precautions, and warnings
                    if 'side effects' in sentence.lower():
                        side_effects.append(str(sentence))
                    elif 'benefits' in sentence.lower():
                        benefits.append(str(sentence))
                    elif 'precautions' in sentence.lower():
                        precautions.append(str(sentence))
                    elif 'warnings' in sentence.lower():
                        warnings.append(str(sentence))

            # Pass the safety information to the HTML template
            return render_template('safety_result.html', 
                                    drug_name=drug_name, 
                                    condition=condition, 
                                    side_effects='\n'.join(side_effects) if side_effects else 'No side effects found',
                                    benefits='\n'.join(benefits) if benefits else 'No benefits found',
                                    precautions='\n'.join(precautions) if precautions else 'No precautions found',
                                    warnings='\n'.join(warnings) if warnings else 'No warnings found')
        else:
            # Drug not found in the dataset
            error_message = f"The drug '{drug_name}' was not found in the dataset."
            return render_template('safety_result.html', error_message=error_message)

    return render_template('safety.html')

# Load the dataset

@app.route('/get_reviews', methods=['POST'])
def get_reviews():
    df = pd.read_csv('data/output.csv')
    
    if request.method == 'POST':
        drug_name = request.form['drug_name']
        drug_reviews = df[df['drugName'] == drug_name][['review', 'sentiment']].values.tolist()
        
        positive_reviews = []
        neutral_reviews = []
        negative_reviews = []
        
        for review, sentiment in drug_reviews:
            if sentiment == 'positive' or sentiment == 'Positive':
                positive_reviews.append(review)
            elif sentiment == 'neutral' or sentiment == 'Neutral':
                neutral_reviews.append(review)
            elif sentiment == 'negative' or sentiment == 'Negative':
                negative_reviews.append(review)
        
        categorized_reviews = {
            'positive': positive_reviews,
            'neutral': neutral_reviews,
            'negative': negative_reviews
        }
        
        return render_template('displayReview.html', drug_name=drug_name, positive_reviews=positive_reviews, neutral_reviews=neutral_reviews, negative_reviews=negative_reviews)

if __name__ == "__main__":
	
	app.run(debug=True, host="localhost", port=8080)