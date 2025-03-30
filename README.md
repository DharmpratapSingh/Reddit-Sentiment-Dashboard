# Reddit-Sentiment-Dashboard

## Abstract

The Reddit Sentiment Dashboard is an interactive web application that performs real-time sentiment and emotion analysis on Reddit posts using multiple state-of-the-art NLP models. Leveraging technologies such as Python, Streamlit, PRAW, and HuggingFace Transformers, the dashboard aggregates data, generates dynamic visualizations, and provides actionable insights. This project demonstrates the integration of advanced machine learning models with data visualization and scalable data ingestion techniques.

## Introduction

In today’s data-rich environment, understanding public sentiment on social media is vital for businesses, researchers, and policymakers. Reddit, with its diverse communities and vast amount of user-generated content, serves as an excellent source for gauging public opinion. The Reddit Sentiment Dashboard was developed to:
	•	Evaluate the sentiment and underlying emotions of Reddit posts.
	•	Compare outputs from four distinct NLP models (Cardiff, DistilBERT, NLPTown, and BERTweet) to ensure robust analysis.
	•	Provide temporal and engagement insights through interactive visualizations.

This project not only showcases advanced NLP techniques but also integrates full-stack development principles using Streamlit to create a user-friendly and scalable application.

## Motivation

The motivation behind this project is twofold:
	1.	Technical Exploration: To harness and compare multiple NLP models in real time, demonstrating the effectiveness of various sentiment analysis techniques.
	2.	Actionable Insights: To provide stakeholders with rapid, interpretable visual insights from social media data, enabling data-driven decision-making.

## Methods

Data Ingestion
	•	Source: Reddit data is acquired using the PRAW library.
	•	Approach: The app fetches posts from user-selected subreddits in real time. Data is then cleaned and structured for analysis.

Sentiment & Emotion Analysis
	•	Multi-Model Sentiment Analysis:
	•	Models Used: Cardiff, DistilBERT, NLPTown, BERTweet
	•	Process: Each model evaluates post content and returns a sentiment label along with a confidence score.
	•	Emotion Detection:
	•	An additional emotion model classifies posts into emotions (e.g., joy, sadness, anger) to provide deeper insights into the text’s affective tone.

Visualization and User Interface
	•	Streamlit Framework: Provides a multi-tab layout for segregating analyses:
	•	Dashboard: Displays individual post-level sentiment and emotion results.
	•	Temporal Analysis: Visualizes trends over time using bar charts, scatter plots, and line graphs.
	•	User Behavior: Presents engagement metrics such as comment sentiment and count.
	•	Model Insights: Compares model performance with radar charts and aggregate metrics.
	•	Dynamic Visualizations: Interactive charts are generated using Plotly and Pandas, allowing users to explore data trends and model performance.

## Results

Sentiment and Emotion Analysis
	•	Model Comparison:
	•	When analyzing posts with multiple models, discrepancies and agreements in sentiment provide a robust overview of public opinion.
	•	Confidence scores are visualized, enabling a quick assessment of each model’s reliability.
	•	Visual Insights:
	•	Temporal Trends: Bar and radar charts help in understanding sentiment distribution over time.
	•	Engagement Metrics: Analysis of comment data adds another layer of insight into user interaction.

Key Achievements
	•	Real-Time Analysis: The dashboard processes Reddit posts in real time, ensuring up-to-date insights.
	•	Scalability: PRAW integration and optimized data workflows allow for handling large volumes of posts and comments.
	•	Actionable Insights: The visualizations and multi-model comparisons empower users to make informed decisions based on sentiment trends.

## Discussion

The multi-model approach allows for cross-validation between different NLP methods, improving overall reliability. The integration of emotion detection adds nuance, revealing not just whether a post is positive or negative but also what kind of emotion it conveys. The dashboard’s design supports rapid exploration of data, making it an effective tool for monitoring public sentiment on Reddit.

## Future Work

Potential enhancements include:
	•	Expanding Model Coverage: Incorporate additional NLP models to further validate sentiment predictions.
	•	Enhanced Temporal Analysis: Introduce more granular time series analyses and forecasting.
	•	User Personalization: Add user-specific filters (e.g., by subreddit, time range, or topic) for more customized insights.
	•	Deployment and Scaling: Optimize the app for deployment on cloud platforms to handle increased load and real-time streaming data.

## Conclusion

The Reddit Sentiment Dashboard effectively demonstrates the fusion of advanced NLP, data visualization, and scalable data ingestion. By comparing multiple sentiment models and visualizing trends, the project delivers actionable insights into Reddit’s dynamic social media landscape. This comprehensive tool not only highlights technical proficiency but also provides a valuable resource for data-driven decision-making.
