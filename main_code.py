import streamlit as st
import praw
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
from functools import wraps
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT
from app.multi_model_sentiment import analyze_sentiment_multi
from app.emotion_detector import analyze_emotion

# Set up page configuration and title
st.set_page_config(page_title="Reddit Sentiment Dashboard", layout="wide")
st.title("🧠 Reddit Multimodel Sentiment & Emotion Analysis Dashboard")


# Rate limiting decorator
def rate_limit(max_calls=30, period=60):
    """
    Decorator to rate limit function calls.
    Default: 30 calls per 60 seconds (Reddit API limit: 60/min)
    """
    def decorator(func):
        calls = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls outside the time window
            calls[:] = [call_time for call_time in calls if call_time > now - period]

            if len(calls) >= max_calls:
                sleep_time = period - (now - calls[0])
                if sleep_time > 0:
                    st.warning(f"⏳ Rate limit approaching. Waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    calls[:] = []

            calls.append(time.time())
            return func(*args, **kwargs)

        return wrapper
    return decorator


# Helper functions for color-coding based on sentiment and emotion labels
def get_sentiment_color(sentiment):
    mapping = {
        "positive": "green",
        "negative": "red",
        "neutral": "gray",
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


# Cached sentiment analysis to avoid re-computation
@st.cache_data(show_spinner=False)
def cached_sentiment_analysis(text):
    """Cache sentiment analysis results to improve performance."""
    return analyze_sentiment_multi(text)


@st.cache_data(show_spinner=False)
def cached_emotion_analysis(text):
    """Cache emotion analysis results to improve performance."""
    return analyze_emotion(text)


# Validate subreddit name
def validate_subreddit(reddit, subreddit_name):
    """Validate that a subreddit exists and is accessible."""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        # Try to access subreddit ID to verify it exists
        _ = subreddit.id
        return True, subreddit
    except Exception as e:
        return False, str(e)


# Fetch posts with error handling and rate limiting
@rate_limit(max_calls=30, period=60)
def fetch_reddit_posts(reddit, subreddit_name, limit=5):
    """
    Fetch posts from Reddit with comprehensive error handling.

    Args:
        reddit: PRAW Reddit instance
        subreddit_name: Name of the subreddit
        limit: Number of posts to fetch

    Returns:
        list: List of post dictionaries or error message
    """
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = []

        for post in subreddit.hot(limit=limit):
            content = (post.title or "") + " " + (post.selftext or "")
            content = content.strip()

            # Skip empty posts
            if not content:
                continue

            timestamp = post.created_utc
            posts.append({
                "content": content,
                "timestamp": timestamp,
                "praw_post": post
            })

        if not posts:
            return {"error": "No posts found with content in this subreddit."}

        return posts

    except Exception as e:
        return {"error": f"Failed to fetch posts: {str(e)}"}


# Initialize Reddit client with error handling
@st.cache_resource
def initialize_reddit_client():
    """Initialize Reddit client with credentials validation."""
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        # Test the connection
        _ = reddit.user.me()
        return reddit, None
    except Exception as e:
        return None, f"Failed to initialize Reddit client: {str(e)}"


# Initialize Reddit client
reddit, init_error = initialize_reddit_client()

if init_error:
    st.error(f"❌ {init_error}")
    st.info("Please check your .env file and ensure Reddit API credentials are set correctly.")
    st.stop()

# Get subreddit settings from the user
try:
    popular_subreddits = [sub.display_name for sub in reddit.subreddits.popular(limit=20)]
except Exception as e:
    st.warning(f"⚠️ Could not fetch popular subreddits: {e}")
    popular_subreddits = ["AskReddit", "technology", "science", "worldnews", "funny"]

# Manual subreddit input option
col1, col2 = st.columns([3, 1])
with col1:
    subreddit_input = st.selectbox("🔍 Choose a subreddit:", popular_subreddits)
with col2:
    custom_subreddit = st.text_input("Or enter custom:", "")

if custom_subreddit:
    subreddit_input = custom_subreddit.strip()

# Validate subreddit
is_valid, validation_result = validate_subreddit(reddit, subreddit_input)
if not is_valid:
    st.error(f"❌ Invalid subreddit '{subreddit_input}': {validation_result}")
    st.stop()

num_posts = st.slider("📄 Number of posts to fetch:", 1, 50, 5)
model_options = ["cardiff", "distilbert", "nlptown", "bertweet", "all"]
selected_model = st.selectbox("🧠 Sentiment model:", model_options, index=0)

# Add a fetch button to control when to fetch posts
if st.button("🚀 Fetch & Analyze Posts", type="primary"):
    with st.spinner(f"Fetching {num_posts} posts from r/{subreddit_input}..."):
        # Fetch posts from the selected subreddit
        posts_result = fetch_reddit_posts(reddit, subreddit_input, num_posts)

        # Check for errors
        if isinstance(posts_result, dict) and "error" in posts_result:
            st.error(f"❌ {posts_result['error']}")
            st.stop()

        posts = posts_result

        if not posts:
            st.warning("No posts found. Try a different subreddit or increase the post limit.")
            st.stop()

        # Store posts in session state
        st.session_state['posts'] = posts
        st.session_state['subreddit'] = subreddit_input
        st.success(f"✅ Successfully fetched {len(posts)} posts from r/{subreddit_input}!")

# Check if posts are available in session state
if 'posts' not in st.session_state:
    st.info("👆 Click 'Fetch & Analyze Posts' to start the analysis!")
    st.stop()

posts = st.session_state['posts']

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

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, post in enumerate(posts, 1):
        post_content = post["content"]
        timestamp = post["timestamp"]

        # Update progress
        progress_bar.progress(idx / len(posts))
        status_text.text(f"Analyzing post {idx}/{len(posts)}...")

        try:
            sentiment_results = cached_sentiment_analysis(post_content)

            # Check for errors in sentiment analysis
            if isinstance(sentiment_results, dict) and "error" in sentiment_results:
                st.warning(f"⚠️ Post {idx}: {sentiment_results['error']}")
                continue

            st.markdown(f"### 🔎 Post {idx}")

            # Display sentiment results (using color-coded text)
            if selected_model == "all":
                for model in ["cardiff", "distilbert", "nlptown", "bertweet"]:
                    if model in sentiment_results:
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
                if selected_model in sentiment_results:
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
            emotion_label, _ = cached_emotion_analysis(post_content)
            st.write(
                f"- **😶 Emotion**: "
                f"<span style='color:{get_emotion_color(emotion_label)}'>{emotion_label.capitalize()}</span>",
                unsafe_allow_html=True
            )

            # Display post content with length indicator
            content_preview = post_content[:300]
            if len(post_content) > 300:
                content_preview += "..."
                st.caption(f"📝 {content_preview} (Total: {len(post_content)} chars)")
            else:
                st.caption(f"📝 {content_preview}")

            st.caption(f"🕒 Timestamp (UTC): {datetime.utcfromtimestamp(timestamp)}")
            st.markdown("---")

            data.append({
                "post_content": post_content,
                "sentiment_label": label_to_visualize,
                "emotion": emotion_label,
                "timestamp": timestamp
            })

        except Exception as e:
            st.error(f"❌ Error analyzing post {idx}: {str(e)}")
            continue

    progress_bar.empty()
    status_text.empty()

    # Store data in session state for other tabs
    st.session_state['analysis_data'] = data

# ---------------------------
# Tab 2: Temporal Analysis
# ---------------------------
with tab2:
    st.markdown("## 📆 Emotion & Sentiment Trends Over Time")

    if 'analysis_data' not in st.session_state or not st.session_state['analysis_data']:
        st.warning("No analysis data available. Please run the analysis in the Dashboard tab first.")
    else:
        data = st.session_state['analysis_data']
        df = pd.DataFrame(data)
        df['datetime'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))

        st.write(f"🗃️ **Posts Data** ({len(df)} posts analyzed)")
        st.dataframe(df)

        # CSV download button for posts DataFrame
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Posts Data as CSV",
            data=csv,
            file_name=f'reddit_sentiment_{st.session_state["subreddit"]}_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

        try:
            fig_sentiment = px.bar(df, x="sentiment_label", color="sentiment_label",
                                  title="Sentiment Distribution",
                                  labels={"sentiment_label": "Sentiment"})
            st.plotly_chart(fig_sentiment, use_container_width=True)

            fig_emotion = px.bar(df, x="emotion", color="emotion",
                                title="Emotion Distribution",
                                labels={"emotion": "Emotion"})
            st.plotly_chart(fig_emotion, use_container_width=True)

            fig_time = px.scatter(df, x="datetime", y="emotion", color="emotion",
                                 title="Emotion Trajectory Over Time",
                                 hover_data=["post_content"],
                                 labels={"datetime": "Time", "emotion": "Emotion"})
            st.plotly_chart(fig_time, use_container_width=True)

            df['date'] = df['datetime'].dt.date
            emotion_over_time = df.groupby(['date', 'emotion']).size().reset_index(name='count')
            fig_line = px.line(emotion_over_time, x="date", y="count", color="emotion",
                              title="Aggregated Emotion Frequency Over Time",
                              markers=True,
                              labels={"date": "Date", "count": "Frequency", "emotion": "Emotion"})
            st.plotly_chart(fig_line, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error creating visualizations: {str(e)}")

# ---------------------------
# Tab 3: User Behavior Insights
# ---------------------------
with tab3:
    st.markdown("## 📊 User Behavior Insights: Comment Engagement")

    if 'posts' not in st.session_state:
        st.warning("No posts available. Please fetch posts first.")
    else:
        engagement_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, post in enumerate(posts, 1):
            progress_bar.progress(idx / len(posts))
            status_text.text(f"Analyzing comments for post {idx}/{len(posts)}...")

            praw_post = post["praw_post"]
            try:
                praw_post.comments.replace_more(limit=0)
                top_comments = praw_post.comments[:3]
                comment_count = len(praw_post.comments.list())
                comment_texts = [c.body for c in top_comments if hasattr(c, 'body') and c.body]

                if comment_texts:
                    sentiments = [cached_sentiment_analysis(c)['cardiff']['label'] for c in comment_texts]
                    avg_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "N/A"
                else:
                    avg_sentiment = "N/A"

            except Exception as e:
                comment_count = 0
                avg_sentiment = "N/A"
                st.caption(f"⚠️ Could not fetch comments for post {idx}: {str(e)}")

            engagement_data.append({
                "post_number": idx,
                "post_content": post["content"][:100] + "...",
                "comment_count": comment_count,
                "avg_comment_sentiment": avg_sentiment
            })

        progress_bar.empty()
        status_text.empty()

        engagement_df = pd.DataFrame(engagement_data)
        st.write("🗃️ **Engagement Data**")
        st.dataframe(engagement_df)

        try:
            fig_engagement = px.bar(engagement_df, x="post_number", y="comment_count",
                                   title="📈 Comment Count per Post",
                                   labels={"post_number": "Post", "comment_count": "Comments"},
                                   hover_data=["post_content"])
            st.plotly_chart(fig_engagement, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating engagement chart: {str(e)}")

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

    if 'posts' not in st.session_state:
        st.warning("No posts available. Please fetch posts first.")
    else:
        # Initialize dictionaries to collect model confidences and agreement scores
        model_confidences = {"cardiff": [], "distilbert": [], "nlptown": [], "bertweet": []}
        model_agreements = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Iterate over posts to compute confidences and agreement per post
        for idx, post in enumerate(posts, 1):
            progress_bar.progress(idx / len(posts))
            status_text.text(f"Computing model insights for post {idx}/{len(posts)}...")

            try:
                sentiment_results = cached_sentiment_analysis(post["content"])

                if isinstance(sentiment_results, dict) and "error" not in sentiment_results:
                    # Append the maximum confidence for each model
                    for model in model_confidences:
                        if model in sentiment_results:
                            model_confidences[model].append(max(sentiment_results[model]['probs']))

                    # Compute agreement: count how many models agree on the sentiment label
                    labels = [sentiment_results[model]['label'] for model in model_confidences if model in sentiment_results]
                    if labels:
                        majority_label = max(set(labels), key=labels.count)
                        agreement = labels.count(majority_label) / len(labels)
                        model_agreements.append(agreement)
            except Exception as e:
                st.caption(f"⚠️ Error computing insights for post {idx}: {str(e)}")
                continue

        progress_bar.empty()
        status_text.empty()

        # Compute average confidence per model
        avg_confidences = {
            model: (sum(vals) / len(vals) if vals else 0)
            for model, vals in model_confidences.items()
        }

        # Compute average model agreement
        avg_agreement = sum(model_agreements) / len(model_agreements) if model_agreements else 0

        st.subheader("Average Confidence per Model")
        conf_df = pd.DataFrame(list(avg_confidences.items()), columns=["Model", "Avg Confidence"])
        st.dataframe(conf_df)
        st.bar_chart(conf_df.set_index("Model"))

        st.subheader("Model Agreement")
        st.metric("Average agreement among models", f"{avg_agreement * 100:.2f}%")

        if avg_agreement > 0.75:
            st.success("✅ High model agreement indicates reliable predictions!")
        elif avg_agreement > 0.5:
            st.info("ℹ️ Moderate model agreement. Results are generally consistent.")
        else:
            st.warning("⚠️ Low model agreement. The posts may contain nuanced or ambiguous sentiment.")

        st.subheader("Radar Chart of Average Model Confidence")
        try:
            import plotly.graph_objects as go

            categories = list(avg_confidences.keys())
            values = list(avg_confidences.values())

            if values:
                # Close the loop for radar chart
                categories.append(categories[0])
                values.append(values[0])

                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Model Confidence'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(values) + 0.1] if values else [0, 1]
                        )
                    ),
                    showlegend=False,
                    title="Radar Chart of Average Model Confidence"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.warning("No data available for radar chart.")
        except Exception as e:
            st.error(f"❌ Radar chart could not be generated: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Reddit Sentiment Dashboard v2.0 | Built with Streamlit, PRAW, and HuggingFace Transformers</p>
    <p>⚡ Powered by 5 state-of-the-art NLP models</p>
</div>
""", unsafe_allow_html=True)
