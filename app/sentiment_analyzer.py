from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# We'll use the 'cardiffnlp/twitter-roberta-base-sentiment' model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABELS = ["negative", "neutral", "positive"]

def analyze_sentiment(text):
    # Tokenize and truncate to avoid overly long text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Convert logits to probabilities
    probs = softmax(outputs.logits.numpy()[0])
    # Pick the label with the highest probability
    sentiment_label = LABELS[probs.argmax()]
    return sentiment_label, probs