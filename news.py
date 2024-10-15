import requests
import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import re
from datetime import datetime
import plotly.express as px  # Import Plotly for pie chart

# Download VADER lexicon (run once)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load the Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # Run on CPU

# Set up NewsAPI key
NEWSAPI_KEY = 'df839167bfdd4ad082f36cc658dd9688'  # Replace with your NewsAPI key

# Function to get articles from NewsAPI
def get_articles_from_newsapi(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()
        return articles['articles']
    else:
        st.error(f"Error: {response.status_code}")
        return []

# Function to extract sentences mentioning the celebrity or brand
def extract_relevant_sentences(text, entity):
    if not text:
        return ""
    
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    relevant_sentences = [sentence for sentence in sentences if entity.lower() in sentence.lower()]
    return ' '.join(relevant_sentences)

# Function to summarize the relevant sentences using Hugging Face's summarizer
def summarize_relevant_text(text, entity):
    relevant_text = extract_relevant_sentences(text, entity)
    
    if not relevant_text:  
        relevant_text = text

    if not relevant_text or len(relevant_text.split()) < 50:  
        return relevant_text if relevant_text else "No relevant text found."

    summary = summarizer(relevant_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Function to perform sentiment analysis on text using NLTK's VADER
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

# Classify articles based on sentiment
def classify_sentiment(compound):
    if compound > 0.05:
        return "Positive"
    elif compound < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Streamlit App Layout
st.title("Sentiment Analysis on NewsAPI Articles")
st.subheader("Search for Celebrities or Brands in NewsAPI Articles and Analyze Sentiment")

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
            sentiment = analyze_sentiment(summary)
            
            article_data.append([
                title, 
                summary, 
                url, 
                sentiment['pos'],  
                sentiment['neu'],  
                sentiment['neg'],  
                sentiment['compound'],  
                published_at  
            ])

        # Create DataFrame
        df = pd.DataFrame(article_data, columns=['Title', 'Summary', 'URL', 'Positive', 'Neutral', 'Negative', 'Compound', 'Published At'])

        # Sort by publication date
        df = df.sort_values(by='Published At', ascending=False)

        # Add sentiment category
        df['Sentiment Category'] = df['Compound'].apply(classify_sentiment)

        # Display DataFrame in Streamlit
        st.write(f"Sentiment analysis for {celebrity_or_brand}")

        # Display articles with links, summaries, sentiment scores, and publication dates
        for index, row in df.iterrows():
            st.markdown(f"### [{row['Title']}]({row['URL']})")
            st.write(f"Published At: {row['Published At']}")
            st.write(f"Summary: {row['Summary']}")
            st.write(f"**Sentiment Breakdown**: Positive: {row['Positive']}, Neutral: {row['Neutral']}, Negative: {row['Negative']}, Compound: {row['Compound']}")
            st.write("---")

        # Sentiment Distribution (Compound Scores)
        st.subheader("Sentiment Distribution (Compound Scores)")
        st.bar_chart(df[['Title', 'Compound']].set_index('Title'))

        # Sentiment Distribution (Pie Chart)
        st.subheader("Sentiment Distribution (Pie Chart)")
        sentiment_distribution = df['Sentiment Category'].value_counts().reset_index()
        sentiment_distribution.columns = ['Sentiment', 'Count']

        # Create pie chart using Plotly
        fig = px.pie(sentiment_distribution, values='Count', names='Sentiment', title="Sentiment Distribution")
        st.plotly_chart(fig)

        # Plot sentiment change over time
        st.subheader("Sentiment Change Over Time")
        df = df.dropna(subset=['Published At'])

        df.set_index('Published At', inplace=True)
        st.line_chart(df['Compound'])

    else:
        st.warning("No articles found for the given search term.")
