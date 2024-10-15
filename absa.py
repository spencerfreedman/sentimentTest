import requests
import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import spacy
import re
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Download necessary data
nltk.download('vader_lexicon')

spacy.cli.download("en_core_web_sm")


# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load the Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# Load Spacy for entity recognition (NER)
nlp = spacy.load("en_core_web_sm")  # Pre-trained NER model

# Set up NewsAPI key

NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]

# Function to get articles from NewsAPI
def get_articles_from_newsapi(query, api_key, page=1):
    url = f"https://newsapi.org/v2/everything?q={query}&page={page}&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        articles = response.json()
        return articles['articles']
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return []

# Function to extract sentences mentioning the celebrity or brand
def extract_relevant_sentences(text, entity):
    if not text:
        return ""
    
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    relevant_sentences = [sentence for sentence in sentences if entity.lower() in sentence.lower()]
    return ' '.join(relevant_sentences)

# Function to perform ABSA using Spacy NER and VADER for sentiment analysis
def aspect_based_sentiment(text):
    doc = nlp(text)  # Apply NER
    aspects = [(ent.text, ent.label_) for ent in doc.ents]  # Extract entities
    aspect_sentiments = {}

    for aspect, label in aspects:
        relevant_sentences = extract_relevant_sentences(text, aspect)
        if relevant_sentences:
            sentiment = sia.polarity_scores(relevant_sentences)
            aspect_sentiments[aspect] = {
                "Sentiment": sentiment['compound'],
                "Positive": sentiment['pos'],
                "Neutral": sentiment['neu'],
                "Negative": sentiment['neg'],
                "Entity Type": label
            }

    return aspect_sentiments

# Function to summarize the relevant sentences using Hugging Face's summarizer
def summarize_relevant_text(text, entity):
    relevant_text = extract_relevant_sentences(text, entity)
    
    if not relevant_text:
        relevant_text = text

    if not relevant_text or len(relevant_text.split()) < 50:
        return relevant_text if relevant_text else "No relevant text found."

    summary = summarizer(relevant_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Plot stacked bar chart for sentiment distribution across articles
def plot_stacked_sentiment_chart(df):
    # Extract data for plotting
    article_titles = df['Title']
    positive = [aspect["Positive"] for article in df["Aspects"] for aspect in article.values()]
    neutral = [aspect["Neutral"] for article in df["Aspects"] for aspect in article.values()]
    negative = [aspect["Negative"] for article in df["Aspects"] for aspect in article.values()]

    # Create stacked bar chart
    fig = go.Figure(data=[
        go.Bar(name='Positive', x=article_titles, y=positive, marker_color='green'),
        go.Bar(name='Neutral', x=article_titles, y=neutral, marker_color='gray'),
        go.Bar(name='Negative', x=article_titles, y=negative, marker_color='red')
    ])

    # Update layout for stacked bars
    fig.update_layout(
        barmode='stack',
        title="Sentiment Distribution Across Articles",
        xaxis_title="Articles",
        yaxis_title="Sentiment Score",
        legend_title="Sentiment Type",
        xaxis_tickangle=-45,  # Tilt article titles for better readability
    )
    
    st.plotly_chart(fig)

# Plot sentiment over time line chart
def plot_sentiment_over_time(df):
    df['Published At'] = pd.to_datetime(df['Published At'])

    # Prepare the data for plotting
    sentiment_time_data = {
        "Published At": [row['Published At'] for index, row in df.iterrows() for aspect in row['Aspects'].values()],
        "Positive": [aspect["Positive"] for row in df["Aspects"] for aspect in row.values()],
        "Neutral": [aspect["Neutral"] for row in df["Aspects"] for aspect in row.values()],
        "Negative": [aspect["Negative"] for row in df["Aspects"] for aspect in row.values()]
    }

    sentiment_df = pd.DataFrame(sentiment_time_data)

    # Create line chart
    fig = px.line(sentiment_df, x="Published At", y=["Positive", "Neutral", "Negative"], 
                  labels={"value": "Sentiment Score", "Published At": "Publication Date"},
                  title="Sentiment Over Time")
    
    fig.update_layout(
        xaxis_title="Publication Date",
        yaxis_title="Sentiment Score",
        legend_title="Sentiment Type"
    )
    
    st.plotly_chart(fig)

# Streamlit App Layout
st.title("Aspect-Based Sentiment Analysis on NewsAPI Articles")
st.subheader("Search for Celebrities or Brands in NewsAPI Articles and Perform ABSA")

# Input field for celebrity or brand name
celebrity_or_brand = st.text_input("Enter the name of a celebrity or brand", value="Taylor Swift")

# Button to trigger the article search
if st.button("Analyze Sentiment"):
    with st.spinner("Fetching articles..."):
        articles = get_articles_from_newsapi(celebrity_or_brand, NEWSAPI_KEY)

    if articles:
        # Prepare data
        article_data = []
        for article in articles:
            title = article.get('title', 'No title')
            description = article.get('description')
            url = article.get('url', 'No URL available')
            published_at = article.get('publishedAt', None)

            if published_at:
                published_at = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')

            if description is None:
                description = "No description available."

            summary = summarize_relevant_text(description, celebrity_or_brand)

            # Perform ABSA for each article
            aspect_sentiments = aspect_based_sentiment(description)

            # Store title, summary, aspects, URL, and publication date
            article_data.append({
                "Title": title,
                "Summary": summary,
                "Aspects": aspect_sentiments,
                "URL": url,
                "Published At": published_at
            })

        # Create DataFrame
        df = pd.DataFrame(article_data)

        # Display DataFrame in Streamlit
        st.write(f"Aspect-based sentiment analysis for {celebrity_or_brand}")

        # Display each article's aspects and sentiment breakdown
        for index, row in df.iterrows():
            st.markdown(f"### [{row['Title']}]({row['URL']})")
            st.write(f"Published At: {row['Published At']}")
            st.write(f"Summary: {row['Summary']}")
            st.write(f"Aspect-Level Sentiments:")
            for aspect, sentiment in row['Aspects'].items():
                st.write(f"**Aspect**: {aspect} (Type: {sentiment['Entity Type']})")
                st.write(f"Sentiment: {sentiment['Sentiment']} (Positive: {sentiment['Positive']}, Neutral: {sentiment['Neutral']}, Negative: {sentiment['Negative']})")
            st.write("---")

        # Plot stacked sentiment chart
        plot_stacked_sentiment_chart(df)

        # Plot sentiment over time chart
        plot_sentiment_over_time(df)

    else:
        st.warning("No articles found for the given search term.")
