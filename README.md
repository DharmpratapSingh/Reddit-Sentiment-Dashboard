# Reddit Sentiment Dashboard v2.0

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Abstract

The Reddit Sentiment Dashboard is an interactive web application that performs real-time sentiment and emotion analysis on Reddit posts using multiple state-of-the-art NLP models. Leveraging technologies such as Python, Streamlit, PRAW, and HuggingFace Transformers, the dashboard aggregates data, generates dynamic visualizations, and provides actionable insights. This project demonstrates the integration of advanced machine learning models with data visualization and scalable data ingestion techniques.

## What's New in v2.0 🚀

- ✅ **Enhanced Security**: Environment-based credential management (no more hardcoded secrets!)
- ✅ **Rate Limiting**: Intelligent API rate limiting to prevent throttling
- ✅ **Performance Caching**: Sentiment analysis results are cached for faster response
- ✅ **Better Error Handling**: Comprehensive error messages and validation
- ✅ **Input Validation**: Subreddit validation and better input sanitization
- ✅ **Progress Indicators**: Real-time progress bars for long operations
- ✅ **Improved UX**: Fetch button control, better visual feedback
- ✅ **Code Consolidation**: Removed duplicate files for easier maintenance

## Features

### 🎯 Core Capabilities

- **Multi-Model Sentiment Analysis**: Compare results from 4 different NLP models
  - Cardiff RoBERTa (3-class: negative/neutral/positive)
  - DistilBERT (binary: positive/negative)
  - NLPTown BERT (5-star rating system)
  - BERTweet (binary sentiment, Twitter-optimized)

- **Emotion Detection**: Identifies 7 distinct emotions
  - Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral

- **Interactive Dashboard**: 4-tab interface
  - 📋 **Dashboard**: Post-level sentiment and emotion analysis
  - 📈 **Temporal Analysis**: Trend visualization over time
  - 💬 **User Behavior**: Comment engagement metrics
  - 🧠 **Model Insights**: Model performance comparison

- **Data Export**: Download analysis results as CSV

## Prerequisites

- Python 3.8 or higher
- Reddit API credentials (free)
- 4GB+ RAM recommended (for transformer models)
- Internet connection

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Reddit-Sentiment-Dashboard.git
cd Reddit-Sentiment-Dashboard
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Initial installation may take 5-10 minutes as it downloads ~1.4GB of dependencies, including transformer models.

### Step 4: Set Up Reddit API Credentials

1. **Get Reddit API credentials** (free):
   - Go to https://www.reddit.com/prefs/apps
   - Click "create app" or "create another app"
   - Fill in the form:
     - **name**: YourAppName
     - **app type**: Select "script"
     - **description**: (optional)
     - **about url**: (optional)
     - **redirect uri**: http://localhost:8080
   - Click "create app"
   - Note your `client_id` (under the app name) and `client_secret`

2. **Create `.env` file** in the project root:

```bash
# Copy the example file
cp .env.example .env
```

3. **Edit `.env`** and add your credentials:

```env
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=YourAppName:v1.0 (by /u/YourRedditUsername)
```

⚠️ **Important**: Never commit your `.env` file to version control. It's already in `.gitignore`.

## Usage

### Running the Application

```bash
streamlit run main_code.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`

### Using the Dashboard

1. **Select Subreddit**: Choose from popular subreddits or enter a custom one
2. **Configure Settings**:
   - Number of posts (1-50)
   - Sentiment model (single or all 4 models)
3. **Click "Fetch & Analyze Posts"**: Start the analysis
4. **Explore Tabs**:
   - View individual post analysis
   - Analyze trends over time
   - Check comment engagement
   - Compare model performance

### Example Use Cases

**Brand Monitoring**:
```
Subreddit: technology
Posts: 25
Model: All
Goal: Track sentiment about your product
```

**Market Research**:
```
Subreddit: investing
Posts: 50
Model: Cardiff
Goal: Gauge market sentiment
```

**Community Health**:
```
Subreddit: mentalhealth
Posts: 20
Model: All
Goal: Analyze emotional trends
```

## Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **API Client** | PRAW | Reddit data ingestion |
| **ML Framework** | PyTorch | Neural network backend |
| **NLP Models** | HuggingFace Transformers | Pre-trained models |
| **Data Processing** | Pandas | Data manipulation |
| **Visualization** | Plotly | Interactive charts |

### Project Structure

```
Reddit-Sentiment-Dashboard/
├── main_code.py              # Main application (v2.0 - enhanced)
├── config.py                 # Configuration (env-based)
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── LICENSE                  # MIT License
└── app/
    ├── multi_model_sentiment.py   # 4-model sentiment analysis
    ├── emotion_detector.py        # Emotion detection
    └── sentiment_analyzer.py      # Legacy single model
```

### Data Flow

```
User Input → Reddit API (PRAW) → Post Fetching
                                      ↓
                          Multi-Model Analysis
                          ↓         ↓         ↓
                    Sentiment   Emotion   Caching
                          ↓         ↓         ↓
                    Pandas DataFrame Processing
                                      ↓
                    Interactive Visualizations
                    (Plotly Charts & Tables)
```

## Model Details

### Sentiment Analysis Models

| Model | Type | Classes | Training Data | Best For |
|-------|------|---------|---------------|----------|
| Cardiff RoBERTa | 3-class | neg/neu/pos | Twitter | Nuanced sentiment |
| DistilBERT | Binary | pos/neg | SST-2 | Clear polarity |
| NLPTown BERT | 5-star | 1-5 stars | Reviews | Rating-style |
| BERTweet | Binary | pos/neg | Twitter | Social media |

### Emotion Detection Model

- **Model**: j-hartmann/emotion-english-distilroberta-base
- **Emotions**: anger, disgust, fear, joy, neutral, sadness, surprise
- **Training**: Multi-domain emotion dataset

## Performance Optimization

### Caching Strategy
- Sentiment and emotion analysis results are cached using Streamlit's `@st.cache_data`
- Reddit client initialization is cached with `@st.cache_resource`
- Significantly reduces computation time for repeated analyses

### Rate Limiting
- Automatic rate limiting prevents Reddit API throttling
- Default: 30 calls per 60 seconds (conservative for safety)
- Visual warnings when approaching limits

### Memory Management
- Models loaded once at startup
- Shared across all analysis requests
- ~1.5-2GB RAM footprint for all 5 models

## Troubleshooting

### Common Issues

**1. "Reddit API credentials not found"**
```
Solution: Ensure .env file exists with correct credentials
Check: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are set
```

**2. "Failed to initialize Reddit client"**
```
Solution: Verify credentials are correct
Check: Reddit app type is "script" not "web app"
```

**3. "Rate limit approaching"**
```
Solution: Wait for the indicated time
Note: This is normal and protects your API quota
```

**4. "Module not found"**
```
Solution: Reinstall dependencies
Command: pip install -r requirements.txt --upgrade
```

**5. Out of memory errors**
```
Solution: Reduce number of posts or close other applications
Requirement: 4GB+ RAM recommended
```

### Debug Mode

Enable verbose logging:
```bash
streamlit run main_code.py --logger.level=debug
```

## API Rate Limits

Reddit API limits:
- **Authenticated**: 60 requests per minute
- **Burst**: 600 requests per 10 minutes

The dashboard implements conservative rate limiting (30 calls/min) to ensure smooth operation.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements

Planned features for v3.0:
- [ ] Database persistence (PostgreSQL/MongoDB)
- [ ] Historical trend analysis
- [ ] Sentiment forecasting (ARIMA/LSTM)
- [ ] Multi-subreddit comparison
- [ ] Email alerts for sentiment thresholds
- [ ] GPU acceleration support
- [ ] Docker containerization
- [ ] RESTful API endpoint
- [ ] User authentication

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **HuggingFace** for providing pre-trained transformer models
- **PRAW** developers for the excellent Reddit API wrapper
- **Streamlit** team for the amazing web framework
- Reddit community for providing valuable data

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{reddit_sentiment_dashboard,
  author = {DharmpratapSingh Vaghela},
  title = {Reddit Sentiment Dashboard: Multi-Model Sentiment Analysis Tool},
  year = {2024},
  url = {https://github.com/yourusername/Reddit-Sentiment-Dashboard},
  version = {2.0}
}
```

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Reddit-Sentiment-Dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Reddit-Sentiment-Dashboard/discussions)

## Version History

### v2.0 (Current)
- Enhanced security with environment variables
- Added rate limiting and error handling
- Implemented performance caching
- Improved user experience
- Code consolidation and cleanup

### v1.0
- Initial release
- Basic multi-model sentiment analysis
- 4-tab dashboard interface
- Temporal and engagement analysis

---

**Made with ❤️ using Python, Streamlit, and Transformers**

⭐ If you find this project useful, please consider giving it a star on GitHub!
