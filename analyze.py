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
    return sentiment_score

# Function to explain sentiment based on individual sentence scores
def explain_sentiment(text, sentiment_type):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)  # Split text into sentences
    sentence_scores = [(sentence, sia.polarity_scores(sentence)['compound']) for sentence in sentences]
    
    if sentiment_type == 'Positive':
        # Extract the most positive sentences
        explanation_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:2]
    elif sentiment_type == 'Negative':
        # Extract the most negative sentences
        explanation_sentences = sorted(sentence_scores, key=lambda x: x[1])[:2]
    else:
        # Extract the most neutral sentences
        explanation_sentences = sorted(sentence_scores, key=lambda x: abs(x[1]))[:2]

    explanation = " ".join([sentence for sentence, score in explanation_sentences])
    return explanation

# Function to highlight mentions of the celebrity/brand
def highlight_mentions(text, target):
    # Use <mark> HTML tag to highlight mentions of the target
    highlighted_text = re.sub(f'({target})', r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return highlighted_text

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
            sentiment_score = analyze_sentiment(summary)
            compound_score = sentiment_score['compound']
            
            # Determine sentiment type based on compound score
            if compound_score >= 0.05:
                sentiment_type = 'Positive'
            elif compound_score <= -0.05:
                sentiment_type = 'Negative'
            else:
                sentiment_type = 'Neutral'

            # Generate explanation for the sentiment
            explanation = explain_sentiment(summary, sentiment_type)
            
            # Highlight mentions of the celebrity/brand in the summary
            highlighted_summary = highlight_mentions(summary, celebrity_or_brand)
            
            article_data.append([headline, highlighted_summary, web_url, sentiment_type, explanation])

        # Create DataFrame
        df = pd.DataFrame(article_data, columns=['Headline', 'Summary', 'URL', 'Sentiment', 'Explanation'])

        # Display DataFrame in Streamlit
        st.write(f"Sentiment analysis for {celebrity_or_brand}")

        # Display articles with links, summaries (highlighted), sentiment, and explanations
        for index, row in df.iterrows():
            st.markdown(f"### [{row['Headline']}]({row['URL']})")
            # Display highlighted summary as HTML
            st.markdown(f"Summary: {row['Summary']}", unsafe_allow_html=True)
            st.write(f"Sentiment: {row['Sentiment']}")
            st.write(f"Explanation: {row['Explanation']}")
            st.write("---")

        # Sentiment Distribution
        sentiment_count = df['Sentiment'].value_counts()
        st.subheader("Sentiment Distribution")
        st.bar_chart(sentiment_count)
    else:
        st.warning("No articles found for the given search term.")
