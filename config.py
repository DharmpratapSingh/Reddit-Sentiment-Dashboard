"""
Configuration file for Reddit API credentials.
Loads credentials from environment variables for security.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Reddit API credentials - loaded from environment variables
CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'RSentimentDashboard:v1.0')

# Validate that credentials are set
if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        "Reddit API credentials not found! "
        "Please create a .env file with REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET. "
        "See .env.example for template."
    )
