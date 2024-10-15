import requests
import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import re

# Download VADER lexicon (run once)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load the Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # Run on CPU

# Set up NYTimes API key
NYTIMES_API_KEY = 'xVuzAYCXHA5yeu7CaJj5MW0I5VyhGuGD'

# Function to get articles from New York Times API
def get_articles(query, api_key):
    url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={query}&api-key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()
        return articles['response']['docs']
    else:
        st.error(f"Error: {response.status_code}")
        return []

# Function to extract sentences mentioning the celebrity or brand
def extract_relevant_sentences(text, target):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # Split text into sentences
    relevant_sentences = [sentence for sentence in sentences if target.lower() in sentence.lower()]
    return ' '.join(relevant_sentences)

# Function to summarize the relevant sentences using Hugging Face's summarizer
def summarize_relevant_text(text, target):
    relevant_text = extract_relevant_sentences(text, target)
    
    if not relevant_text:  # If no relevant text, summarize the whole snippet as fallback
        relevant_text = text

    if len(relevant_text.split()) < 50:  # If the relevant text is already short, no need to summarize
        return relevant_text
    
    summary = summarizer(relevant_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Function to perform sentiment analysis on text using NLTK's VADER
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit App Layout
st.title("NYTimes Articles Sentiment Analysis on Celebrities/Brands")
st.subheader("Search for Celebrities or Brands in NYTimes Articles, Summarize, and Analyze Sentiment")

# Input field for celebrity or brand name
celebrity_or_brand = st.text_input("Enter the name of a celebrity or brand", value="Taylor Swift")

# Button to trigger the article search
if st.button("Analyze Sentiment"):
    with st.spinner("Fetching articles..."):
        articles = get_articles(celebrity_or_brand, NYTIMES_API_KEY)

    if articles:
        # Prepare data
        article_data = []
        for article in articles:
            headline = article.get('headline', {}).get('main', 'No headline')
            snippet = article.get('snippet', 'No snippet available')
            web_url = article.get('web_url', 'No URL')  # Get the URL for the article
            
            # Summarize the relevant part of the snippet
            summary = summarize_relevant_text(snippet, celebrity_or_brand)
            
            # Analyze sentiment on the summary
            sentiment = analyze_sentiment(summary)
            
            article_data.append([headline, summary, web_url, sentiment])

        # Create DataFrame
        df = pd.DataFrame(article_data, columns=['Headline', 'Summary', 'URL', 'Sentiment'])

        # Display DataFrame in Streamlit
        st.write(f"Sentiment analysis for {celebrity_or_brand}")

        # Display articles with links and summaries
        for index, row in df.iterrows():
            st.markdown(f"### [{row['Headline']}]({row['URL']})")
            st.write(f"Summary: {row['Summary']}")
            st.write(f"Sentiment: {row['Sentiment']}")
            st.write("---")

        # Sentiment Distribution
        sentiment_count = df['Sentiment'].value_counts()
        st.subheader("Sentiment Distribution")
        st.bar_chart(sentiment_count)
    else:
        st.warning("No articles found for the given search term.")
