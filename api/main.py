"""
FastAPI REST API for Reddit Sentiment Dashboard.
Provides programmatic access to sentiment analysis functionality.
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.multi_model_sentiment import analyze_sentiment_multi
from app.emotion_detector import analyze_emotion
from app.advanced_nlp import advanced_nlp
from app.database import DatabaseManager
import praw
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT


# Initialize FastAPI app
app = FastAPI(
    title="Reddit Sentiment Analysis API",
    description="REST API for analyzing sentiment and emotions in Reddit posts",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = DatabaseManager()

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)


# Pydantic models for request/response
class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1)
    models: Optional[List[str]] = Field(
        default=["cardiff"],
        description="List of sentiment models to use"
    )
    include_emotion: bool = Field(default=True, description="Include emotion analysis")
    include_nlp: bool = Field(default=False, description="Include advanced NLP features")


class SubredditAnalysisRequest(BaseModel):
    """Request model for subreddit analysis."""
    subreddit: str = Field(..., description="Subreddit name")
    limit: int = Field(default=10, ge=1, le=100, description="Number of posts to analyze")
    models: Optional[List[str]] = Field(
        default=["cardiff"],
        description="Sentiment models to use"
    )
    include_emotion: bool = Field(default=True, description="Include emotion analysis")
    include_nlp: bool = Field(default=False, description="Include advanced NLP")
    save_to_db: bool = Field(default=True, description="Save results to database")


class SubredditComparisonRequest(BaseModel):
    """Request model for comparing multiple subreddits."""
    subreddits: List[str] = Field(..., description="List of subreddit names", min_items=2, max_items=5)
    limit: int = Field(default=10, ge=1, le=50, description="Posts per subreddit")
    start_date: Optional[datetime] = Field(default=None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(default=None, description="End date for filtering")


# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check."""
    return {
        "status": "online",
        "message": "Reddit Sentiment Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "analyze_text": "/api/v1/analyze/text",
            "analyze_subreddit": "/api/v1/analyze/subreddit",
            "compare_subreddits": "/api/v1/compare",
            "database_stats": "/api/v1/stats"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    db_stats = db.get_database_stats()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": {
            "connected": True,
            "stats": db_stats
        },
        "reddit_api": {
            "configured": bool(CLIENT_ID and CLIENT_SECRET)
        }
    }


@app.post("/api/v1/analyze/text", tags=["Analysis"])
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze sentiment and emotion of a single text.

    Args:
        request: TextAnalysisRequest with text and options

    Returns:
        Analysis results including sentiment, emotion, and optional NLP features
    """
    try:
        # Sentiment analysis
        sentiment_results = analyze_sentiment_multi(request.text)

        if isinstance(sentiment_results, dict) and "error" in sentiment_results:
            raise HTTPException(status_code=400, detail=sentiment_results["error"])

        # Filter models if specified
        if request.models and request.models != ["all"]:
            sentiment_results = {
                k: v for k, v in sentiment_results.items()
                if k in request.models
            }

        response = {
            "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "sentiment": sentiment_results
        }

        # Emotion analysis
        if request.include_emotion:
            emotion_label, emotion_probs = analyze_emotion(request.text)
            response["emotion"] = {
                "label": emotion_label,
                "probabilities": emotion_probs
            }

        # Advanced NLP
        if request.include_nlp:
            nlp_analysis = advanced_nlp.comprehensive_analysis(request.text)
            response["nlp"] = nlp_analysis

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/subreddit", tags=["Analysis"])
async def analyze_subreddit(request: SubredditAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze posts from a subreddit.

    Args:
        request: SubredditAnalysisRequest with subreddit name and options

    Returns:
        Analysis results for multiple posts
    """
    try:
        # Fetch posts from Reddit
        subreddit = reddit.subreddit(request.subreddit)
        posts_data = []

        for post in subreddit.hot(limit=request.limit):
            content = (post.title or "") + " " + (post.selftext or "")
            content = content.strip()

            if not content:
                continue

            post_data = {
                'post_id': post.id,
                'subreddit': request.subreddit,
                'title': post.title,
                'content': content,
                'author': str(post.author) if post.author else None,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'url': post.url
            }

            # Analyze sentiment
            sentiment_results = analyze_sentiment_multi(content)

            if isinstance(sentiment_results, dict) and "error" not in sentiment_results:
                # Filter models
                if request.models and request.models != ["all"]:
                    sentiment_results = {
                        k: v for k, v in sentiment_results.items()
                        if k in request.models
                    }

                post_data['sentiment'] = sentiment_results

                # Emotion analysis
                if request.include_emotion:
                    emotion_label, emotion_probs = analyze_emotion(content)
                    post_data['emotion'] = {
                        'label': emotion_label,
                        'probabilities': emotion_probs
                    }

                # Advanced NLP
                if request.include_nlp:
                    nlp_analysis = advanced_nlp.comprehensive_analysis(content)
                    post_data['nlp'] = nlp_analysis

                # Save to database
                if request.save_to_db:
                    post_db_id = db.insert_post(post_data)

                    if post_db_id:
                        # Save sentiment results
                        for model_name, result in sentiment_results.items():
                            db.insert_sentiment(
                                post_db_id,
                                model_name,
                                result['label'],
                                max(result['probs']),
                                result['probs']
                            )

                        # Save emotion
                        if request.include_emotion:
                            db.insert_emotion(
                                post_db_id,
                                emotion_label,
                                max(emotion_probs),
                                emotion_probs
                            )

                        # Save NLP entities
                        if request.include_nlp and 'entities' in nlp_analysis:
                            db.insert_named_entities(post_db_id, nlp_analysis['entities'])

            posts_data.append(post_data)

        return {
            "subreddit": request.subreddit,
            "posts_analyzed": len(posts_data),
            "posts": posts_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/compare", tags=["Comparison"])
async def compare_subreddits(request: SubredditComparisonRequest):
    """
    Compare sentiment across multiple subreddits.

    Args:
        request: SubredditComparisonRequest with list of subreddits

    Returns:
        Comparison data for all subreddits
    """
    try:
        comparison_data = db.compare_subreddits(
            request.subreddits,
            request.start_date,
            request.end_date
        )

        return {
            "subreddits": request.subreddits,
            "comparison": comparison_data,
            "filters": {
                "start_date": request.start_date.isoformat() if request.start_date else None,
                "end_date": request.end_date.isoformat() if request.end_date else None
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/subreddit/{subreddit}/stats", tags=["Statistics"])
async def get_subreddit_stats(
    subreddit: str,
    start_date: Optional[datetime] = Query(default=None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(default=None, description="End date for filtering")
):
    """
    Get statistics for a specific subreddit.

    Args:
        subreddit: Subreddit name
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        Statistics including sentiment and emotion distributions
    """
    try:
        sentiment_stats = db.get_sentiment_statistics(subreddit, start_date, end_date)
        emotion_stats = db.get_emotion_statistics(subreddit, start_date, end_date)
        posts = db.get_posts_by_subreddit(subreddit, start_date, end_date)

        return {
            "subreddit": subreddit,
            "post_count": len(posts),
            "sentiment_distribution": sentiment_stats,
            "emotion_distribution": emotion_stats,
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/subreddit/{subreddit}/entities", tags=["NLP"])
async def get_trending_entities(
    subreddit: str,
    entity_type: Optional[str] = Query(default=None, description="Filter by entity type"),
    limit: int = Query(default=10, ge=1, le=50, description="Number of entities to return")
):
    """
    Get trending named entities in a subreddit.

    Args:
        subreddit: Subreddit name
        entity_type: Optional entity type filter (PERSON, ORG, LOC)
        limit: Number of top entities

    Returns:
        List of trending entities with counts
    """
    try:
        entities = db.get_trending_entities(subreddit, entity_type, limit)

        return {
            "subreddit": subreddit,
            "entity_type": entity_type,
            "trending_entities": [
                {"text": entity[0], "count": entity[1]}
                for entity in entities
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats", tags=["Statistics"])
async def get_database_stats():
    """
    Get overall database statistics.

    Returns:
        Database statistics including post counts, subreddits tracked, etc.
    """
    try:
        stats = db.get_database_stats()
        return {
            "database_stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models", tags=["Information"])
async def list_models():
    """
    List available sentiment and emotion models.

    Returns:
        List of available models
    """
    return {
        "sentiment_models": {
            "cardiff": {
                "name": "Cardiff RoBERTa",
                "description": "3-class sentiment (negative/neutral/positive)",
                "training_data": "Twitter"
            },
            "distilbert": {
                "name": "DistilBERT SST-2",
                "description": "Binary sentiment (positive/negative)",
                "training_data": "SST-2 dataset"
            },
            "nlptown": {
                "name": "NLPTown BERT",
                "description": "5-star rating system",
                "training_data": "Reviews (multilingual)"
            },
            "bertweet": {
                "name": "BERTweet",
                "description": "Binary sentiment (positive/negative)",
                "training_data": "Twitter"
            }
        },
        "emotion_model": {
            "name": "Emotion RoBERTa",
            "description": "7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise)",
            "model": "j-hartmann/emotion-english-distilroberta-base"
        },
        "nlp_features": {
            "ner": "Named Entity Recognition (PERSON, ORG, LOC, MISC)",
            "keywords": "Keyword extraction",
            "sarcasm": "Sarcasm detection",
            "readability": "Readability analysis (Flesch score)"
        }
    }


@app.delete("/api/v1/data/cleanup", tags=["Maintenance"])
async def cleanup_old_data(days: int = Query(default=30, ge=1, le=365, description="Days of data to keep")):
    """
    Delete posts older than specified days.

    Args:
        days: Number of days to keep

    Returns:
        Number of posts deleted
    """
    try:
        deleted_count = db.delete_old_data(days)

        return {
            "status": "success",
            "posts_deleted": deleted_count,
            "days_retained": days
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
