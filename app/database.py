"""
Database module for Reddit Sentiment Dashboard.
Handles SQLite operations for storing and retrieving analysis data.
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
import os


class DatabaseManager:
    """Manage SQLite database operations for sentiment analysis data."""

    def __init__(self, db_path: str = "sentiment_data.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Posts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id TEXT UNIQUE,
                    subreddit TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    author TEXT,
                    score INTEGER,
                    num_comments INTEGER,
                    post_created_utc DATETIME,
                    fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    url TEXT
                )
            """)

            # Sentiment results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER,
                    model_name TEXT NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    confidence FLOAT,
                    probabilities TEXT,
                    analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
                )
            """)

            # Emotions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER,
                    emotion_label TEXT NOT NULL,
                    confidence FLOAT,
                    probabilities TEXT,
                    analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
                )
            """)

            # Named entities table (for NER)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS named_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER,
                    entity_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    confidence FLOAT,
                    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
                )
            """)

            # Topics table (for topic modeling)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER,
                    topic_id INTEGER,
                    topic_label TEXT,
                    probability FLOAT,
                    keywords TEXT,
                    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_subreddit
                ON posts(subreddit)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_created
                ON posts(post_created_utc)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_model
                ON sentiment_results(model_name)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_label
                ON sentiment_results(sentiment_label)
            """)

    def insert_post(self, post_data: Dict) -> Optional[int]:
        """
        Insert a post into the database.

        Args:
            post_data: Dictionary containing post information

        Returns:
            Post ID if successful, None otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    INSERT INTO posts (
                        post_id, subreddit, title, content, author,
                        score, num_comments, post_created_utc, url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_data.get('post_id'),
                    post_data.get('subreddit'),
                    post_data.get('title'),
                    post_data.get('content'),
                    post_data.get('author'),
                    post_data.get('score'),
                    post_data.get('num_comments'),
                    post_data.get('created_utc'),
                    post_data.get('url')
                ))
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Post already exists, get its ID
                cursor.execute("SELECT id FROM posts WHERE post_id = ?",
                             (post_data.get('post_id'),))
                result = cursor.fetchone()
                return result[0] if result else None

    def insert_sentiment(self, post_db_id: int, model_name: str,
                        sentiment_label: str, confidence: float,
                        probabilities: List[float]) -> bool:
        """
        Insert sentiment analysis result.

        Args:
            post_db_id: Database ID of the post
            model_name: Name of the sentiment model
            sentiment_label: Predicted sentiment label
            confidence: Confidence score
            probabilities: List of probability scores

        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sentiment_results (
                    post_id, model_name, sentiment_label, confidence, probabilities
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                post_db_id, model_name, sentiment_label,
                confidence, json.dumps(probabilities)
            ))
            return True

    def insert_emotion(self, post_db_id: int, emotion_label: str,
                      confidence: float, probabilities: List[float]) -> bool:
        """
        Insert emotion analysis result.

        Args:
            post_db_id: Database ID of the post
            emotion_label: Predicted emotion label
            confidence: Confidence score
            probabilities: List of probability scores

        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO emotions (
                    post_id, emotion_label, confidence, probabilities
                ) VALUES (?, ?, ?, ?)
            """, (
                post_db_id, emotion_label, confidence, json.dumps(probabilities)
            ))
            return True

    def insert_named_entities(self, post_db_id: int, entities: List[Dict]) -> bool:
        """
        Insert named entities for a post.

        Args:
            post_db_id: Database ID of the post
            entities: List of entity dictionaries

        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for entity in entities:
                cursor.execute("""
                    INSERT INTO named_entities (
                        post_id, entity_text, entity_type, confidence
                    ) VALUES (?, ?, ?, ?)
                """, (
                    post_db_id,
                    entity.get('text'),
                    entity.get('type'),
                    entity.get('confidence', 0.0)
                ))
            return True

    def insert_topics(self, post_db_id: int, topics: List[Dict]) -> bool:
        """
        Insert topic modeling results for a post.

        Args:
            post_db_id: Database ID of the post
            topics: List of topic dictionaries

        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for topic in topics:
                cursor.execute("""
                    INSERT INTO topics (
                        post_id, topic_id, topic_label, probability, keywords
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    post_db_id,
                    topic.get('topic_id'),
                    topic.get('label'),
                    topic.get('probability'),
                    json.dumps(topic.get('keywords', []))
                ))
            return True

    def get_posts_by_subreddit(self, subreddit: str,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               limit: int = 100) -> List[Dict]:
        """
        Retrieve posts from a specific subreddit with optional date filtering.

        Args:
            subreddit: Name of the subreddit
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            limit: Maximum number of posts to return

        Returns:
            List of post dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM posts WHERE subreddit = ?"
            params = [subreddit]

            if start_date:
                query += " AND post_created_utc >= ?"
                params.append(start_date)

            if end_date:
                query += " AND post_created_utc <= ?"
                params.append(end_date)

            query += " ORDER BY post_created_utc DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_sentiment_by_post(self, post_db_id: int) -> List[Dict]:
        """
        Get all sentiment results for a post.

        Args:
            post_db_id: Database ID of the post

        Returns:
            List of sentiment result dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sentiment_results
                WHERE post_id = ?
                ORDER BY analyzed_at DESC
            """, (post_db_id,))
            results = [dict(row) for row in cursor.fetchall()]

            # Parse JSON probabilities
            for result in results:
                if result.get('probabilities'):
                    result['probabilities'] = json.loads(result['probabilities'])

            return results

    def get_emotion_by_post(self, post_db_id: int) -> Dict:
        """
        Get emotion result for a post.

        Args:
            post_db_id: Database ID of the post

        Returns:
            Emotion result dictionary
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM emotions
                WHERE post_id = ?
                ORDER BY analyzed_at DESC
                LIMIT 1
            """, (post_db_id,))
            result = cursor.fetchone()

            if result:
                result_dict = dict(result)
                if result_dict.get('probabilities'):
                    result_dict['probabilities'] = json.loads(result_dict['probabilities'])
                return result_dict

            return {}

    def get_sentiment_statistics(self, subreddit: str,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Dict:
        """
        Get sentiment statistics for a subreddit.

        Args:
            subreddit: Name of the subreddit
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary with sentiment statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    sr.sentiment_label,
                    COUNT(*) as count,
                    AVG(sr.confidence) as avg_confidence,
                    sr.model_name
                FROM sentiment_results sr
                JOIN posts p ON sr.post_id = p.id
                WHERE p.subreddit = ?
            """
            params = [subreddit]

            if start_date:
                query += " AND p.post_created_utc >= ?"
                params.append(start_date)

            if end_date:
                query += " AND p.post_created_utc <= ?"
                params.append(end_date)

            query += " GROUP BY sr.sentiment_label, sr.model_name"

            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]

            return results

    def get_emotion_statistics(self, subreddit: str,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict:
        """
        Get emotion statistics for a subreddit.

        Args:
            subreddit: Name of the subreddit
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary with emotion statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    e.emotion_label,
                    COUNT(*) as count,
                    AVG(e.confidence) as avg_confidence
                FROM emotions e
                JOIN posts p ON e.post_id = p.id
                WHERE p.subreddit = ?
            """
            params = [subreddit]

            if start_date:
                query += " AND p.post_created_utc >= ?"
                params.append(start_date)

            if end_date:
                query += " AND p.post_created_utc <= ?"
                params.append(end_date)

            query += " GROUP BY e.emotion_label"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_named_entities_by_post(self, post_db_id: int) -> List[Dict]:
        """
        Get named entities for a post.

        Args:
            post_db_id: Database ID of the post

        Returns:
            List of entity dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM named_entities
                WHERE post_id = ?
            """, (post_db_id,))
            return [dict(row) for row in cursor.fetchall()]

    def compare_subreddits(self, subreddits: List[str],
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict:
        """
        Compare sentiment across multiple subreddits.

        Args:
            subreddits: List of subreddit names
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary with comparison data
        """
        comparison = {}

        for subreddit in subreddits:
            comparison[subreddit] = {
                'sentiment': self.get_sentiment_statistics(subreddit, start_date, end_date),
                'emotions': self.get_emotion_statistics(subreddit, start_date, end_date),
                'post_count': len(self.get_posts_by_subreddit(subreddit, start_date, end_date))
            }

        return comparison

    def get_trending_entities(self, subreddit: str,
                             entity_type: Optional[str] = None,
                             limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get most common named entities in a subreddit.

        Args:
            subreddit: Name of the subreddit
            entity_type: Optional filter by entity type (PERSON, ORG, LOC, etc.)
            limit: Number of top entities to return

        Returns:
            List of (entity_text, count) tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT ne.entity_text, COUNT(*) as count
                FROM named_entities ne
                JOIN posts p ON ne.post_id = p.id
                WHERE p.subreddit = ?
            """
            params = [subreddit]

            if entity_type:
                query += " AND ne.entity_type = ?"
                params.append(entity_type)

            query += " GROUP BY ne.entity_text ORDER BY count DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [(row['entity_text'], row['count']) for row in cursor.fetchall()]

    def delete_old_data(self, days: int = 30) -> int:
        """
        Delete posts older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of posts deleted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM posts
                WHERE post_created_utc < datetime('now', '-' || ? || ' days')
            """, (days,))
            return cursor.rowcount

    def get_database_stats(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Count posts
            cursor.execute("SELECT COUNT(*) FROM posts")
            stats['total_posts'] = cursor.fetchone()[0]

            # Count sentiments
            cursor.execute("SELECT COUNT(*) FROM sentiment_results")
            stats['total_sentiments'] = cursor.fetchone()[0]

            # Count emotions
            cursor.execute("SELECT COUNT(*) FROM emotions")
            stats['total_emotions'] = cursor.fetchone()[0]

            # Count entities
            cursor.execute("SELECT COUNT(*) FROM named_entities")
            stats['total_entities'] = cursor.fetchone()[0]

            # Database size
            stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0

            # Subreddits tracked
            cursor.execute("SELECT COUNT(DISTINCT subreddit) FROM posts")
            stats['subreddits_tracked'] = cursor.fetchone()[0]

            return stats
