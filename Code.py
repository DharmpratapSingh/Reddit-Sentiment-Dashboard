import streamlit as st
import praw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT
from app.sentiment_analyzer import analyze_sentiment

# Set up the Streamlit app
st.title("Reddit Sentiment Analysis Dashboard")

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# 1) Fetch posts from a subreddit
# Retrieve a list of popular subreddits (you can adjust the limit as needed)
popular_subreddits = [sub.display_name for sub in reddit.subreddits.popular(limit=20)]
subreddit_input = st.selectbox("Choose a subreddit:", popular_subreddits, index=0)
subreddit = reddit.subreddit(subreddit_input)
num_posts = st.slider("Select number of posts to fetch:", min_value=1, max_value=50, value=5)
posts = []
for post in subreddit.hot(limit=num_posts):
    content = (post.title or "") + " " + (post.selftext or "")
    posts.append(content.strip())

# 2) Analyze sentiment for each post and display results
st.header("Post Sentiment Analysis")
for idx, post_content in enumerate(posts, start=1):
    label, probs = analyze_sentiment(post_content)
    st.write(f"**Post {idx} Sentiment:** {label} (probs={probs})")
    st.write(f"**Post Content:** {post_content}")
    st.write("---")

# Prepare data for visualization
data = []
for post_content in posts:
    label, probs = analyze_sentiment(post_content)
    data.append({
        "post_content": post_content,
        "sentiment_label": label,
        "negative_prob": probs[0],
        "neutral_prob": probs[1],
        "positive_prob": probs[2]
    })

df = pd.DataFrame(data)
st.write("### DataFrame of Posts with Sentiment", df)

# Create and display a sentiment distribution plot using Seaborn
st.subheader("Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(x="sentiment_label", data=df, ax=ax, palette="viridis")
ax.set_title("Sentiment Distribution")
st.pyplot(fig)