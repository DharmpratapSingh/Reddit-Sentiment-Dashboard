"""
Advanced NLP features for Reddit Sentiment Dashboard.
Includes Named Entity Recognition, Topic Modeling, and more.
"""
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Tuple, Optional
import re
import streamlit as st


class AdvancedNLP:
    """Advanced NLP processing for text analysis."""

    def __init__(self):
        """Initialize NLP models."""
        self._ner_pipeline = None
        self._zero_shot_classifier = None

    @st.cache_resource
    def _load_ner_model(_self):
        """Load Named Entity Recognition model."""
        try:
            return pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
        except Exception as e:
            print(f"Error loading NER model: {e}")
            return None

    @st.cache_resource
    def _load_zero_shot_classifier(_self):
        """Load zero-shot classification model."""
        try:
            return pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        except Exception as e:
            print(f"Error loading zero-shot classifier: {e}")
            return None

    def extract_named_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text using NER.

        Args:
            text: Input text

        Returns:
            List of entity dictionaries with text, type, and confidence
        """
        if self._ner_pipeline is None:
            self._ner_pipeline = self._load_ner_model()

        if self._ner_pipeline is None:
            return []

        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            entities = self._ner_pipeline(text)

            # Format results
            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    'text': entity['word'],
                    'type': entity['entity_group'],
                    'confidence': float(entity['score']),
                    'start': entity['start'],
                    'end': entity['end']
                })

            return formatted_entities

        except Exception as e:
            print(f"Error in NER: {e}")
            return []

    def categorize_entities(self, entities: List[Dict]) -> Dict[str, List[str]]:
        """
        Group entities by type.

        Args:
            entities: List of entity dictionaries

        Returns:
            Dictionary with entity types as keys and lists of entity texts
        """
        categorized = {
            'PERSON': [],
            'ORG': [],
            'LOC': [],
            'MISC': []
        }

        for entity in entities:
            entity_type = entity.get('type', 'MISC')
            entity_text = entity.get('text', '')

            if entity_type in categorized:
                if entity_text not in categorized[entity_type]:
                    categorized[entity_type].append(entity_text)
            else:
                if entity_text not in categorized['MISC']:
                    categorized['MISC'].append(entity_text)

        return categorized

    def classify_topic(self, text: str, candidate_labels: List[str]) -> Dict:
        """
        Classify text into predefined topics using zero-shot classification.

        Args:
            text: Input text
            candidate_labels: List of possible topic labels

        Returns:
            Dictionary with labels and their probabilities
        """
        if self._zero_shot_classifier is None:
            self._zero_shot_classifier = self._load_zero_shot_classifier()

        if self._zero_shot_classifier is None:
            return {}

        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            result = self._zero_shot_classifier(
                text,
                candidate_labels,
                multi_label=False
            )

            return {
                'labels': result['labels'],
                'scores': result['scores']
            }

        except Exception as e:
            print(f"Error in topic classification: {e}")
            return {}

    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords from text using simple statistical methods.

        Args:
            text: Input text
            top_n: Number of top keywords to return

        Returns:
            List of (keyword, score) tuples
        """
        # Simple keyword extraction using word frequency
        # In production, consider using YAKE, KeyBERT, or similar

        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())

        # Split into words
        words = text.split()

        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't'
        }

        # Filter words
        words = [w for w in words if w not in stop_words and len(w) > 2]

        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Return top N
        return sorted_words[:top_n]

    def detect_sarcasm(self, text: str) -> Dict:
        """
        Detect if text contains sarcasm (simple rule-based approach).

        Args:
            text: Input text

        Returns:
            Dictionary with sarcasm detection result
        """
        # Simple rule-based sarcasm detection
        # In production, use a trained sarcasm detection model

        sarcasm_indicators = [
            'yeah right',
            'sure thing',
            'oh really',
            'wow so',
            'great job',
            'nice try',
            'obviously',
            'clearly',
            'totally'
        ]

        text_lower = text.lower()

        # Check for indicators
        indicator_count = sum(1 for indicator in sarcasm_indicators if indicator in text_lower)

        # Check for excessive punctuation (!!!, ???)
        excessive_punct = len(re.findall(r'[!?]{2,}', text))

        # Check for ALL CAPS words
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        caps_ratio = caps_words / len(words) if words else 0

        # Simple scoring
        sarcasm_score = (indicator_count * 0.3) + (excessive_punct * 0.2) + (caps_ratio * 0.5)

        is_sarcastic = sarcasm_score > 0.3

        return {
            'is_sarcastic': is_sarcastic,
            'confidence': min(sarcasm_score, 1.0),
            'indicators_found': indicator_count,
            'excessive_punctuation': excessive_punct > 0,
            'caps_ratio': caps_ratio
        }

    def analyze_readability(self, text: str) -> Dict:
        """
        Analyze text readability.

        Args:
            text: Input text

        Returns:
            Dictionary with readability metrics
        """
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        # Count words
        words = text.split()
        word_count = len(words)

        # Count syllables (simple approximation)
        def count_syllables(word):
            word = word.lower()
            vowels = 'aeiou'
            syllable_count = 0
            previous_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = is_vowel

            # Adjust for silent 'e'
            if word.endswith('e'):
                syllable_count -= 1

            # Ensure at least 1 syllable
            if syllable_count == 0:
                syllable_count = 1

            return syllable_count

        syllable_count = sum(count_syllables(word) for word in words)

        # Calculate metrics
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        avg_syllables_per_word = syllable_count / word_count if word_count > 0 else 0

        # Flesch Reading Ease Score
        # Higher score = easier to read (0-100 scale)
        if sentence_count > 0 and word_count > 0:
            flesch_score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
            flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
        else:
            flesch_score = 0

        # Determine reading level
        if flesch_score >= 90:
            reading_level = "Very Easy"
        elif flesch_score >= 80:
            reading_level = "Easy"
        elif flesch_score >= 70:
            reading_level = "Fairly Easy"
        elif flesch_score >= 60:
            reading_level = "Standard"
        elif flesch_score >= 50:
            reading_level = "Fairly Difficult"
        elif flesch_score >= 30:
            reading_level = "Difficult"
        else:
            reading_level = "Very Difficult"

        return {
            'sentence_count': sentence_count,
            'word_count': word_count,
            'avg_words_per_sentence': round(avg_words_per_sentence, 2),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2),
            'flesch_reading_ease': round(flesch_score, 2),
            'reading_level': reading_level
        }

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text.

        Args:
            text: Input text

        Returns:
            List of URLs found
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls

    def extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text.

        Args:
            text: Input text

        Returns:
            List of hashtags found
        """
        hashtag_pattern = r'#\w+'
        hashtags = re.findall(hashtag_pattern, text)
        return hashtags

    def extract_mentions(self, text: str) -> List[str]:
        """
        Extract user mentions from text.

        Args:
            text: Input text

        Returns:
            List of mentions found
        """
        mention_pattern = r'@\w+'
        mentions = re.findall(mention_pattern, text)
        return mentions

    def comprehensive_analysis(self, text: str) -> Dict:
        """
        Perform comprehensive NLP analysis on text.

        Args:
            text: Input text

        Returns:
            Dictionary with all analysis results
        """
        return {
            'entities': self.extract_named_entities(text),
            'keywords': self.extract_keywords(text),
            'sarcasm': self.detect_sarcasm(text),
            'readability': self.analyze_readability(text),
            'urls': self.extract_urls(text),
            'hashtags': self.extract_hashtags(text),
            'mentions': self.extract_mentions(text)
        }


# Initialize global instance
advanced_nlp = AdvancedNLP()
