# Quick Start Guide - New Features

## 🚀 What's New in v3.0

You now have **5 major new features**:

1. ✅ **SQLite Database** - Persistent storage of all analysis
2. ✅ **Time Range Filtering** - Analyze posts from specific date ranges
3. ✅ **Multi-Subreddit Comparison** - Compare up to 5 subreddits side-by-side
4. ✅ **Advanced NLP** - Named Entity Recognition, keywords, sarcasm detection
5. ✅ **REST API** - Programmatic access via FastAPI

---

## Installation

### 1. Update Dependencies

```bash
pip install -r requirements.txt
```

**New packages installed**:
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

---

## Using the Features

### 1. SQLite Database (Automatic)

The database is created automatically when you run the app!

**Location**: `sentiment_data.db` in your project root

**What's stored**:
- All Reddit posts analyzed
- Sentiment results from all 4 models
- Emotion detection results
- Named entities (people, organizations, locations)
- Topic keywords

**View database stats**:
```python
from app.database import DatabaseManager

db = DatabaseManager()
stats = db.get_database_stats()
print(stats)
```

---

### 2. REST API

#### Start the API Server

```bash
# In terminal
cd /path/to/Reddit-Sentiment-Dashboard
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Access Interactive Documentation

Open in browser: **http://localhost:8000/docs**

You'll see Swagger UI with all endpoints!

#### Quick Test

```bash
# Test health check
curl http://localhost:8000/health

# Analyze text
curl -X POST "http://localhost:8000/api/v1/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love this product!",
    "models": ["cardiff"],
    "include_emotion": true
  }'
```

#### Python Example

```python
import requests

# Analyze subreddit via API
response = requests.post(
    "http://localhost:8000/api/v1/analyze/subreddit",
    json={
        "subreddit": "technology",
        "limit": 10,
        "models": ["cardiff"],
        "include_emotion": True,
        "save_to_db": True
    }
)

data = response.json()
print(f"Analyzed {data['posts_analyzed']} posts")
```

---

### 3. Advanced NLP Features

#### Named Entity Recognition

Automatically extracts:
- **PERSON**: Names of people
- **ORG**: Organizations, companies
- **LOC**: Locations, places
- **MISC**: Miscellaneous entities

**Example**:
```python
from app.advanced_nlp import advanced_nlp

text = "Apple CEO Tim Cook announced new iPhone in California"
entities = advanced_nlp.extract_named_entities(text)

for entity in entities:
    print(f"{entity['text']} ({entity['type']}): {entity['confidence']:.2f}")

# Output:
# Apple (ORG): 0.95
# Tim Cook (PERSON): 0.98
# iPhone (MISC): 0.87
# California (LOC): 0.92
```

#### Keyword Extraction

```python
text = "The new electric vehicle technology is revolutionary..."
keywords = advanced_nlp.extract_keywords(text, top_n=5)

print(keywords)
# [('vehicle', 3), ('technology', 2), ('revolutionary', 1), ...]
```

#### Sarcasm Detection

```python
text = "Oh great, another bug. Just what I needed."
result = advanced_nlp.detect_sarcasm(text)

print(f"Is sarcastic: {result['is_sarcastic']}")
print(f"Confidence: {result['confidence']:.2f}")
```

#### Readability Analysis

```python
text = "Your text here..."
readability = advanced_nlp.analyze_readability(text)

print(f"Reading level: {readability['reading_level']}")
print(f"Flesch score: {readability['flesch_reading_ease']}")
```

---

### 4. Multi-Subreddit Comparison (via API)

Compare sentiment across multiple subreddits:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/compare",
    json={
        "subreddits": ["technology", "science", "futurology"],
        "limit": 20
    }
)

comparison = response.json()

for subreddit, data in comparison['comparison'].items():
    print(f"\nr/{subreddit}:")
    print(f"  Posts: {data['post_count']}")

    for sentiment in data['sentiment']:
        label = sentiment['sentiment_label']
        count = sentiment['count']
        confidence = sentiment['avg_confidence']
        print(f"  {label}: {count} posts (confidence: {confidence:.2f})")
```

**Output**:
```
r/technology:
  Posts: 50
  positive: 25 posts (confidence: 0.85)
  negative: 15 posts (confidence: 0.78)
  neutral: 10 posts (confidence: 0.72)

r/science:
  Posts: 48
  positive: 30 posts (confidence: 0.88)
  ...
```

---

### 5. Time Range Filtering (via API)

Analyze posts from specific date ranges:

```python
import requests
from datetime import datetime, timedelta

# Last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

response = requests.get(
    "http://localhost:8000/api/v1/subreddit/technology/stats",
    params={
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
)

stats = response.json()
print(f"Posts in last 7 days: {stats['post_count']}")
print(f"Sentiment distribution: {stats['sentiment_distribution']}")
```

---

## Common Workflows

### Workflow 1: Analyze & Store in Database

```python
import praw
from app.database import DatabaseManager
from app.multi_model_sentiment import analyze_sentiment_multi
from app.emotion_detector import analyze_emotion
from app.advanced_nlp import advanced_nlp
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT
from datetime import datetime

# Initialize
db = DatabaseManager()
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Fetch posts
subreddit = reddit.subreddit("technology")
for post in subreddit.hot(limit=10):
    content = post.title + " " + post.selftext

    # Insert post
    post_db_id = db.insert_post({
        'post_id': post.id,
        'subreddit': 'technology',
        'title': post.title,
        'content': content,
        'author': str(post.author),
        'score': post.score,
        'num_comments': post.num_comments,
        'created_utc': datetime.fromtimestamp(post.created_utc),
        'url': post.url
    })

    # Analyze sentiment
    sentiment_results = analyze_sentiment_multi(content)

    for model_name, result in sentiment_results.items():
        db.insert_sentiment(
            post_db_id,
            model_name,
            result['label'],
            max(result['probs']),
            result['probs']
        )

    # Analyze emotion
    emotion_label, emotion_probs = analyze_emotion(content)
    db.insert_emotion(post_db_id, emotion_label, max(emotion_probs), emotion_probs)

    # Extract entities
    entities = advanced_nlp.extract_named_entities(content)
    db.insert_named_entities(post_db_id, entities)

    print(f"✅ Stored: {post.title[:50]}...")

print(f"\n📊 Database stats: {db.get_database_stats()}")
```

---

### Workflow 2: Trending Entities

```python
from app.database import DatabaseManager

db = DatabaseManager()

# Get trending organizations in technology subreddit
orgs = db.get_trending_entities("technology", entity_type="ORG", limit=10)

print("Top 10 Organizations mentioned in r/technology:")
for entity_text, count in orgs:
    print(f"  {entity_text}: {count} mentions")
```

---

### Workflow 3: Compare Subreddit Sentiment

```python
from app.database import DatabaseManager

db = DatabaseManager()

# Compare multiple subreddits
comparison = db.compare_subreddits(["technology", "science", "futurology"])

for subreddit, data in comparison.items():
    print(f"\nr/{subreddit}:")
    print(f"  Total posts: {data['post_count']}")

    # Sentiment breakdown
    for sentiment in data['sentiment']:
        print(f"  {sentiment['sentiment_label']}: {sentiment['count']}")
```

---

## API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/api/v1/analyze/text` | POST | Analyze single text |
| `/api/v1/analyze/subreddit` | POST | Analyze subreddit posts |
| `/api/v1/compare` | POST | Compare subreddits |
| `/api/v1/subreddit/{name}/stats` | GET | Get subreddit statistics |
| `/api/v1/subreddit/{name}/entities` | GET | Get trending entities |
| `/api/v1/stats` | GET | Database statistics |
| `/api/v1/models` | GET | List available models |

---

## Running Both Streamlit & API

### Option 1: Two Terminals

**Terminal 1** (Streamlit):
```bash
streamlit run main_code.py
```

**Terminal 2** (API):
```bash
python -m uvicorn api.main:app --reload --port 8000
```

### Option 2: Docker Compose (Coming Soon)

We'll add docker-compose support to run both services together!

---

## Database Management

### View Database Contents

```python
from app.database import DatabaseManager

db = DatabaseManager()

# Get all posts from a subreddit
posts = db.get_posts_by_subreddit("technology", limit=10)

for post in posts:
    print(f"- {post['title']}")

    # Get sentiment for this post
    sentiments = db.get_sentiment_by_post(post['id'])
    for sent in sentiments:
        print(f"  {sent['model_name']}: {sent['sentiment_label']}")
```

### Clean Old Data

```python
from app.database import DatabaseManager

db = DatabaseManager()

# Delete posts older than 30 days
deleted = db.delete_old_data(days=30)
print(f"Deleted {deleted} old posts")
```

---

## Troubleshooting

### Issue: API won't start

**Solution**:
```bash
# Check if port 8000 is in use
netstat -tulpn | grep 8000

# Use different port
uvicorn api.main:app --port 8001
```

### Issue: Database locked

**Solution**:
```python
# Close any open connections
# Delete sentiment_data.db and restart
```

### Issue: NER model not loading

**Solution**:
```bash
# Model will download on first use
# If failing, check internet connection
# Models are ~500MB, may take time
```

---

## Performance Tips

1. **Use caching**: Sentiment results are cached automatically
2. **Batch processing**: Use API for batch analysis
3. **Database indexes**: Already optimized
4. **Limit posts**: Start with small limits (10-20 posts)
5. **Background processing**: Use API background tasks

---

## Next Steps

1. ✅ **Try the API**: http://localhost:8000/docs
2. ✅ **Run analysis**: Store some posts in database
3. ✅ **Compare subreddits**: Use comparison endpoint
4. ✅ **Extract entities**: See trending topics
5. ✅ **Build automation**: Schedule regular analysis

---

## Resources

- **API Documentation**: `API_DOCUMENTATION.md`
- **Full Roadmap**: `ROADMAP.md`
- **Deployment Guide**: `DEPLOYMENT.md`
- **Contributing**: `CONTRIBUTING.md`

---

## Examples Repository

Check the `examples/` directory (coming soon) for:
- Complete Python scripts
- Jupyter notebooks
- Integration examples
- Automation scripts

---

**Need Help?** Open an issue on GitHub!

**Have Ideas?** Check ROADMAP.md for planned features!

**Want to Contribute?** Read CONTRIBUTING.md!
