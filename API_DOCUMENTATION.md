
# Reddit Sentiment Dashboard - API Documentation

## Overview

The Reddit Sentiment Dashboard REST API provides programmatic access to sentiment analysis, emotion detection, and advanced NLP features for Reddit content.

**Base URL**: `http://localhost:8000`

**API Version**: 2.0.0

**Interactive Documentation**: http://localhost:8000/docs (Swagger UI)

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
4. [Request/Response Examples](#examples)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)

---

## Getting Started

### Running the API

```bash
# Start the API server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or with specific workers
uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Interactive docs
open http://localhost:8000/docs
```

---

## Authentication

Currently, the API does not require authentication. In production, implement:
- API keys
- JWT tokens
- OAuth 2.0

### Future Authentication Example

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/api/v1/analyze/text
```

---

## Endpoints

### Health & Information

#### GET `/`
Root endpoint - API health check

**Response**:
```json
{
  "status": "online",
  "message": "Reddit Sentiment Analysis API",
  "version": "2.0.0",
  "endpoints": {...}
}
```

#### GET `/health`
Detailed health check with database statistics

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00",
  "database": {
    "connected": true,
    "stats": {
      "total_posts": 1250,
      "total_sentiments": 5000,
      "subreddits_tracked": 15
    }
  }
}
```

#### GET `/api/v1/models`
List available sentiment and NLP models

**Response**:
```json
{
  "sentiment_models": {
    "cardiff": {...},
    "distilbert": {...},
    "nlptown": {...},
    "bertweet": {...}
  },
  "emotion_model": {...},
  "nlp_features": {...}
}
```

---

### Analysis Endpoints

#### POST `/api/v1/analyze/text`
Analyze sentiment and emotion of a single text

**Request Body**:
```json
{
  "text": "I absolutely love this product! It's amazing!",
  "models": ["cardiff", "distilbert"],
  "include_emotion": true,
  "include_nlp": true
}
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| text | string | Yes | - | Text to analyze |
| models | array | No | ["cardiff"] | Sentiment models to use |
| include_emotion | boolean | No | true | Include emotion analysis |
| include_nlp | boolean | No | false | Include NLP features |

**Response**:
```json
{
  "text": "I absolutely love this product!...",
  "sentiment": {
    "cardiff": {
      "label": "positive",
      "probs": [0.05, 0.10, 0.85]
    },
    "distilbert": {
      "label": "positive",
      "probs": [0.10, 0.90]
    }
  },
  "emotion": {
    "label": "joy",
    "probabilities": [0.02, 0.03, 0.01, 0.88, 0.01, 0.03, 0.02]
  },
  "nlp": {
    "entities": [
      {"text": "product", "type": "MISC", "confidence": 0.92}
    ],
    "keywords": [["love", 2], ["amazing", 1]],
    "sarcasm": {
      "is_sarcastic": false,
      "confidence": 0.15
    },
    "readability": {
      "flesch_reading_ease": 85.5,
      "reading_level": "Easy"
    }
  }
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is the best day ever!",
    "models": ["cardiff"],
    "include_emotion": true,
    "include_nlp": false
  }'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/analyze/text",
    json={
        "text": "This is amazing!",
        "models": ["cardiff", "distilbert"],
        "include_emotion": True,
        "include_nlp": True
    }
)

data = response.json()
print(f"Sentiment: {data['sentiment']['cardiff']['label']}")
print(f"Emotion: {data['emotion']['label']}")
```

---

#### POST `/api/v1/analyze/subreddit`
Analyze posts from a subreddit

**Request Body**:
```json
{
  "subreddit": "technology",
  "limit": 20,
  "models": ["cardiff"],
  "include_emotion": true,
  "include_nlp": false,
  "save_to_db": true
}
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| subreddit | string | Yes | - | Subreddit name |
| limit | integer | No | 10 | Number of posts (1-100) |
| models | array | No | ["cardiff"] | Sentiment models |
| include_emotion | boolean | No | true | Include emotions |
| include_nlp | boolean | No | false | Include NLP |
| save_to_db | boolean | No | true | Save to database |

**Response**:
```json
{
  "subreddit": "technology",
  "posts_analyzed": 20,
  "posts": [
    {
      "post_id": "abc123",
      "title": "New AI breakthrough",
      "content": "Scientists have...",
      "author": "user123",
      "score": 1500,
      "sentiment": {...},
      "emotion": {...}
    }
  ]
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/subreddit" \
  -H "Content-Type: application/json" \
  -d '{
    "subreddit": "technology",
    "limit": 10,
    "models": ["cardiff"],
    "include_emotion": true
  }'
```

---

### Comparison Endpoints

#### POST `/api/v1/compare`
Compare sentiment across multiple subreddits

**Request Body**:
```json
{
  "subreddits": ["technology", "science", "futurology"],
  "limit": 10,
  "start_date": "2025-01-01T00:00:00",
  "end_date": "2025-01-15T23:59:59"
}
```

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| subreddits | array | Yes | 2-5 subreddit names |
| limit | integer | No | Posts per subreddit (1-50) |
| start_date | datetime | No | Start date filter |
| end_date | datetime | No | End date filter |

**Response**:
```json
{
  "subreddits": ["technology", "science"],
  "comparison": {
    "technology": {
      "sentiment": [
        {"sentiment_label": "positive", "count": 45, "avg_confidence": 0.87}
      ],
      "emotions": [
        {"emotion_label": "joy", "count": 30}
      ],
      "post_count": 100
    },
    "science": {...}
  }
}
```

**Python Example**:
```python
import requests
from datetime import datetime, timedelta

response = requests.post(
    "http://localhost:8000/api/v1/compare",
    json={
        "subreddits": ["technology", "science", "futurology"],
        "limit": 20,
        "start_date": (datetime.now() - timedelta(days=7)).isoformat(),
        "end_date": datetime.now().isoformat()
    }
)

comparison = response.json()
for subreddit, data in comparison['comparison'].items():
    print(f"{subreddit}: {data['post_count']} posts")
```

---

### Statistics Endpoints

#### GET `/api/v1/subreddit/{subreddit}/stats`
Get statistics for a specific subreddit

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| subreddit | string | Yes | Subreddit name (path) |
| start_date | datetime | No | Start date (query) |
| end_date | datetime | No | End date (query) |

**Example**:
```bash
curl "http://localhost:8000/api/v1/subreddit/technology/stats?start_date=2025-01-01T00:00:00&end_date=2025-01-15T23:59:59"
```

**Response**:
```json
{
  "subreddit": "technology",
  "post_count": 150,
  "sentiment_distribution": [
    {
      "sentiment_label": "positive",
      "count": 80,
      "avg_confidence": 0.85,
      "model_name": "cardiff"
    }
  ],
  "emotion_distribution": [
    {"emotion_label": "joy", "count": 45, "avg_confidence": 0.82}
  ]
}
```

#### GET `/api/v1/subreddit/{subreddit}/entities`
Get trending named entities in a subreddit

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| subreddit | string | Yes | Subreddit name |
| entity_type | string | No | Filter by type (PERSON, ORG, LOC) |
| limit | integer | No | Number of entities (1-50) |

**Example**:
```bash
curl "http://localhost:8000/api/v1/subreddit/technology/entities?entity_type=ORG&limit=10"
```

**Response**:
```json
{
  "subreddit": "technology",
  "entity_type": "ORG",
  "trending_entities": [
    {"text": "Apple", "count": 45},
    {"text": "Google", "count": 38},
    {"text": "Microsoft", "count": 32}
  ]
}
```

#### GET `/api/v1/stats`
Get overall database statistics

**Response**:
```json
{
  "database_stats": {
    "total_posts": 5000,
    "total_sentiments": 20000,
    "total_emotions": 5000,
    "total_entities": 15000,
    "db_size_mb": 125.5,
    "subreddits_tracked": 25
  },
  "timestamp": "2025-01-15T10:30:00"
}
```

---

### Maintenance Endpoints

#### DELETE `/api/v1/data/cleanup`
Delete posts older than specified days

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| days | integer | No | 30 | Days of data to keep (1-365) |

**Example**:
```bash
curl -X DELETE "http://localhost:8000/api/v1/data/cleanup?days=30"
```

**Response**:
```json
{
  "status": "success",
  "posts_deleted": 150,
  "days_retained": 30
}
```

---

## Error Handling

The API uses standard HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Endpoint doesn't exist |
| 500 | Internal Server Error |

**Error Response Format**:
```json
{
  "error": "Error description",
  "detail": "Detailed error message"
}
```

**Example Errors**:

**400 Bad Request**:
```json
{
  "error": "Invalid input",
  "detail": "Text field is required"
}
```

**500 Internal Server Error**:
```json
{
  "error": "Internal server error",
  "detail": "Database connection failed"
}
```

---

## Rate Limiting

Current implementation has no rate limiting. For production:

**Recommended Limits**:
- 100 requests per minute per IP
- 1000 requests per hour per API key
- Burst allowance: 10 requests

**Rate Limit Headers** (future):
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642253400
```

---

## Complete Usage Examples

### Example 1: Analyze Single Text

```python
import requests

API_URL = "http://localhost:8000"

def analyze_text(text):
    response = requests.post(
        f"{API_URL}/api/v1/analyze/text",
        json={
            "text": text,
            "models": ["cardiff", "distilbert"],
            "include_emotion": True,
            "include_nlp": True
        }
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Sentiment (Cardiff): {data['sentiment']['cardiff']['label']}")
        print(f"Sentiment (DistilBERT): {data['sentiment']['distilbert']['label']}")
        print(f"Emotion: {data['emotion']['label']}")
        print(f"Keywords: {data['nlp']['keywords'][:5]}")
    else:
        print(f"Error: {response.json()}")

# Usage
analyze_text("This is the best day of my life!")
```

### Example 2: Analyze Subreddit and Save to Database

```python
import requests

def analyze_subreddit(subreddit, limit=10):
    response = requests.post(
        "http://localhost:8000/api/v1/analyze/subreddit",
        json={
            "subreddit": subreddit,
            "limit": limit,
            "models": ["cardiff"],
            "include_emotion": True,
            "include_nlp": False,
            "save_to_db": True
        }
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Analyzed {data['posts_analyzed']} posts from r/{subreddit}")

        for post in data['posts']:
            sentiment = post['sentiment']['cardiff']['label']
            emotion = post['emotion']['label']
            print(f"- {post['title'][:50]}... | {sentiment} | {emotion}")
    else:
        print(f"Error: {response.json()}")

# Usage
analyze_subreddit("technology", limit=20)
```

### Example 3: Compare Multiple Subreddits

```python
import requests
from datetime import datetime, timedelta

def compare_subreddits(subreddits, days_back=7):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    response = requests.post(
        "http://localhost:8000/api/v1/compare",
        json={
            "subreddits": subreddits,
            "limit": 10,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    )

    if response.status_code == 200:
        data = response.json()

        for subreddit, stats in data['comparison'].items():
            print(f"\n{subreddit}:")
            print(f"  Posts: {stats['post_count']}")

            # Sentiment distribution
            for sent in stats['sentiment']:
                print(f"  {sent['sentiment_label']}: {sent['count']}")

    else:
        print(f"Error: {response.json()}")

# Usage
compare_subreddits(["technology", "science", "futurology"], days_back=7)
```

### Example 4: Get Trending Entities

```python
import requests

def get_trending_entities(subreddit, entity_type=None, limit=10):
    params = {"limit": limit}
    if entity_type:
        params["entity_type"] = entity_type

    response = requests.get(
        f"http://localhost:8000/api/v1/subreddit/{subreddit}/entities",
        params=params
    )

    if response.status_code == 200:
        data = response.json()
        print(f"\nTrending entities in r/{subreddit}:")

        for entity in data['trending_entities']:
            print(f"  {entity['text']}: {entity['count']} mentions")
    else:
        print(f"Error: {response.json()}")

# Usage
get_trending_entities("technology", entity_type="ORG", limit=10)
```

### Example 5: Monitor Subreddit Sentiment Over Time

```python
import requests
import time
from datetime import datetime

def monitor_sentiment(subreddit, interval_minutes=10):
    """Monitor subreddit sentiment every N minutes."""
    while True:
        response = requests.get(
            f"http://localhost:8000/api/v1/subreddit/{subreddit}/stats"
        )

        if response.status_code == 200:
            data = response.json()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n[{timestamp}] r/{subreddit} Sentiment:")

            for sent in data['sentiment_distribution']:
                print(f"  {sent['sentiment_label']}: {sent['count']} posts")

        time.sleep(interval_minutes * 60)

# Usage
monitor_sentiment("wallstreetbets", interval_minutes=5)
```

---

## JavaScript/Node.js Examples

### Example: Analyze Text with Axios

```javascript
const axios = require('axios');

async function analyzeText(text) {
  try {
    const response = await axios.post('http://localhost:8000/api/v1/analyze/text', {
      text: text,
      models: ['cardiff', 'distilbert'],
      include_emotion: true,
      include_nlp: true
    });

    const data = response.data;
    console.log(`Sentiment: ${data.sentiment.cardiff.label}`);
    console.log(`Emotion: ${data.emotion.label}`);
    console.log(`Keywords:`, data.nlp.keywords.slice(0, 5));

  } catch (error) {
    console.error('Error:', error.response.data);
  }
}

// Usage
analyzeText("This is amazing!");
```

---

## Testing with Postman

1. **Import Collection**: Create a new collection in Postman
2. **Add Requests**: Add the endpoints above
3. **Set Base URL**: Use `http://localhost:8000` as base
4. **Test**: Send requests and verify responses

---

## Deployment

### Docker Deployment

```dockerfile
# Add to Dockerfile
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

1. **Add Authentication**: Implement API keys or JWT
2. **Rate Limiting**: Use Redis + slowapi
3. **Monitoring**: Add logging and metrics
4. **HTTPS**: Use reverse proxy (Nginx)
5. **Caching**: Cache frequent queries
6. **Database**: Use PostgreSQL for production

---

## Support

- **Issues**: https://github.com/yourusername/Reddit-Sentiment-Dashboard/issues
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

**API Version**: 2.0.0
**Last Updated**: 2025-01-15
