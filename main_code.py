import streamlit as st
import praw
import pandas as pd
import plotly.express as px
from datetime import datetime
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT
from app.multi_model_sentiment import analyze_sentiment_multi
from app.emotion_detector import analyze_emotion

# Set up page configuration and title
st.set_page_config(page_title="Reddit Sentiment Dashboard", layout="wide")
st.title("🧠 Reddit Multimodel Sentiment & Emotion Analysis Dashboard")


# Helper functions for color-coding based on sentiment and emotion labels
def get_sentiment_color(sentiment):
    mapping = {
        "positive": "green",
        "negative": "red",
        "neutral": "gray",
        # Additional mappings if needed
    }
    return mapping.get(sentiment.lower(), "black")


def get_emotion_color(emotion):
    mapping = {
        "joy": "#FFD700",  # Gold
        "sadness": "#1E90FF",  # DodgerBlue
        "anger": "#FF4500",  # OrangeRed
        "fear": "#8B008B",  # DarkMagenta
        "surprise": "#32CD32",  # LimeGreen
        "disgust": "#556B2F"  # DarkOliveGreen
    }
    return mapping.get(emotion.lower(), "black")


# Set up Reddit client
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Get subreddit settings from the user
popular_subreddits = [sub.display_name for sub in reddit.subreddits.popular(limit=20)]
subreddit_input = st.selectbox("🔍 Choose a subreddit:", popular_subreddits)
num_posts = st.slider("📄 Number of posts to fetch:", 1, 50, 5)
model_options = ["cardiff", "distilbert", "nlptown", "bertweet", "all"]
selected_model = st.selectbox("🧠 Sentiment model:", model_options, index=0)

# Fetch posts from the selected subreddit
posts = []
for post in reddit.subreddit(subreddit_input).hot(limit=num_posts):
    content = (post.title or "") + " " + (post.selftext or "")
    timestamp = post.created_utc
    posts.append({
        "content": content.strip(),
        "timestamp": timestamp,
        "praw_post": post
    })

# Create a multi-tab layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Dashboard",
    "📈 Temporal Analysis",
    "💬 User Behavior",
    "🧠 Model Insights"
])

# ---------------------------
# Tab 1: Dashboard
# ---------------------------
with tab1:
    st.markdown("## 🧾 Post-Level Sentiment & Emotion Analysis")
    data = []
    for idx, post in enumerate(posts, 1):
        post_content = post["content"]
        timestamp = post["timestamp"]
        sentiment_results = analyze_sentiment_multi(post_content)

        st.markdown(f"### 🔎 Post {idx}")
        # Display sentiment results (using color-coded text)
        if selected_model == "all":
            for model in ["cardiff", "distilbert", "nlptown", "bertweet"]:
                sentiment = sentiment_results[model]['label']
                confidence = max(sentiment_results[model]['probs'])
                st.write(
                    f"- **{model.capitalize()}**: "
                    f"<span style='color:{get_sentiment_color(sentiment)}'>{sentiment.capitalize()}</span> "
                    f"(Confidence: {confidence:.2f})",
                    unsafe_allow_html=True
                )
            label_to_visualize = sentiment_results['cardiff']['label']
        else:
            sentiment = sentiment_results[selected_model]['label']
            confidence = max(sentiment_results[selected_model]['probs'])
            label_to_visualize = sentiment
            st.write(
                f"- **{selected_model.capitalize()}**: "
                f"<span style='color:{get_sentiment_color(sentiment)}'>{sentiment.capitalize()}</span> "
                f"(Confidence: {confidence:.2f})",
                unsafe_allow_html=True
            )

        # Display emotion result (using color-coded text)
        emotion_label, _ = analyze_emotion(post_content)
        st.write(
            f"- **😶 Emotion**: "
            f"<span style='color:{get_emotion_color(emotion_label)}'>{emotion_label.capitalize()}</span>",
            unsafe_allow_html=True
        )
        st.caption(f"📝 {post_content[:300]}...")
        st.caption(f"🕒 Timestamp (UTC): {datetime.utcfromtimestamp(timestamp)}")
        st.markdown("---")

        data.append({
            "post_content": post_content,
            "sentiment_label": label_to_visualize,
            "emotion": emotion_label,
            "timestamp": timestamp
        })

# ---------------------------
# Tab 2: Temporal Analysis
# ---------------------------
with tab2:
    st.markdown("## 📆 Emotion & Sentiment Trends Over Time")
    df = pd.DataFrame(data)
    df['datetime'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    st.write("🗃️ **Posts Data**", df)

    # CSV download button for posts DataFrame
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Posts Data as CSV",
        data=csv,
        file_name='posts_data.csv',
        mime='text/csv'
    )

    fig_sentiment = px.bar(df, x="sentiment_label", color="sentiment_label", title="Sentiment Distribution")
    st.plotly_chart(fig_sentiment)

    fig_emotion = px.bar(df, x="emotion", color="emotion", title="Emotion Distribution")
    st.plotly_chart(fig_emotion)

    fig_time = px.scatter(df, x="datetime", y="emotion", color="emotion",
                          title="Emotion Trajectory Over Time", hover_data=["post_content"])
    st.plotly_chart(fig_time)

    df['date'] = df['datetime'].dt.date
    emotion_over_time = df.groupby(['date', 'emotion']).size().reset_index(name='count')
    fig_line = px.line(emotion_over_time, x="date", y="count", color="emotion",
                       title="Aggregated Emotion Frequency Over Time", markers=True)
    st.plotly_chart(fig_line)

# ---------------------------
# Tab 3: User Behavior Insights
# ---------------------------
with tab3:
    st.markdown("## 📊 User Behavior Insights: Comment Engagement")
    engagement_data = []
    for idx, post in enumerate(posts, 1):
        praw_post = post["praw_post"]
        try:
            praw_post.comments.replace_more(limit=0)
            top_comments = praw_post.comments[:3]
            comment_count = len(praw_post.comments.list())
            comment_texts = [c.body for c in top_comments if c.body]
            sentiments = [analyze_sentiment_multi(c)['cardiff']['label'] for c in comment_texts]
            avg_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "N/A"
        except Exception as e:
            comment_count = 0
            avg_sentiment = "N/A"
        engagement_data.append({
            "post_number": idx,
            "post_content": post["content"],
            "comment_count": comment_count,
            "avg_comment_sentiment": avg_sentiment
        })
    engagement_df = pd.DataFrame(engagement_data)
    st.write("🗃️ **Engagement Data**", engagement_df)
    fig_engagement = px.bar(engagement_df, x="post_number", y="comment_count",
                            title="📈 Comment Count per Post",
                            labels={"post_number": "Post", "comment_count": "Comments"})
    st.plotly_chart(fig_engagement)

# ---------------------------
# Tab 4: Model Insights
# ---------------------------
with tab4:
    st.markdown("## 🧠 Model Insights & Exploration")
    st.markdown("""
    This section includes:
    - **Average Confidence** per model across posts  
    - **Model Agreement**: How often models agree on sentiment  
    - A **Radar Chart** comparing average confidence values
    """)

    # Initialize dictionaries to collect model confidences and agreement scores
    model_confidences = {"cardiff": [], "distilbert": [], "nlptown": [], "bertweet": []}
    model_agreements = []  # list to store agreement percentages for each post

    # Iterate over posts to compute confidences and agreement per post
    for post in posts:
        sentiment_results = analyze_sentiment_multi(post["content"])
        # Append the maximum confidence for each model
        for model in model_confidences:
            model_confidences[model].append(max(sentiment_results[model]['probs']))
        # Compute agreement: count how many models agree on the sentiment label
        labels = [sentiment_results[model]['label'] for model in model_confidences]
        majority_label = max(set(labels), key=labels.count)
        agreement = labels.count(majority_label) / len(labels)
        model_agreements.append(agreement)

    # Compute average confidence per model
    avg_confidences = {model: sum(vals) / len(vals) for model, vals in model_confidences.items()}
    # Compute average model agreement
    avg_agreement = sum(model_agreements) / len(model_agreements) if model_agreements else 0

    st.subheader("Average Confidence per Model")
    conf_df = pd.DataFrame(list(avg_confidences.items()), columns=["Model", "Avg Confidence"])
    st.dataframe(conf_df)
    st.bar_chart(conf_df.set_index("Model"))

    st.subheader("Model Agreement")
    st.write(f"Average agreement among models: {avg_agreement * 100:.2f}%")

    st.subheader("Radar Chart of Average Model Confidence")
    try:
        import plotly.graph_objects as go

        categories = list(avg_confidences.keys())
        values = list(avg_confidences.values())
        # Close the loop for radar chart
        categories.append(categories[0])
        values.append(values[0])
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) + 0.1]
                )
            ),
            showlegend=False,
            title="Radar Chart of Average Model Confidence"
        )
        st.plotly_chart(fig_radar)
    except Exception as e:
        st.write("Radar chart could not be generated.", e)