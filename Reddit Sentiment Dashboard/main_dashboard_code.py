import streamlit as st
import pandas as pd
import praw
import matplotlib.pyplot as plt
import seaborn as sns
from app.sentiment_analyzer import analyze_sentiment
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT

st.title("Reddit Sentiment Analysis Dashboard")

# User inputs
subreddit_input = st.text_input("Enter a subreddit:", "technology")
num_posts = st.slider("Number of posts to fetch:", min_value=5, max_value=50, value=10)

if st.button("Fetch & Analyze"):
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )
    subreddit = reddit.subreddit(subreddit_input)
    posts = []
    for post in subreddit.hot(limit=num_posts):
        content = (post.title or "") + " " + (post.selftext or "")
        posts.append(content.strip())

    data = []
    for post_content in posts:
        label, probs = analyze_sentiment(post_content)
        data.append({
            "Post": post_content,
            "Sentiment": label,
            "Negative": probs[0],
            "Neutral": probs[1],
            "Positive": probs[2]
        })

    df = pd.DataFrame(data)
    st.write("### Fetched Posts and Their Sentiment", df)

    # Plot sentiment distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Sentiment", data=df, ax=ax, palette="viridis")
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)