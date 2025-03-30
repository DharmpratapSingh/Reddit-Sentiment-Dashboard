# app/multi_model_sentiment.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Existing models:
MODELS = {
    "cardiff": "cardiffnlp/twitter-roberta-base-sentiment",   # outputs: negative, neutral, positive
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",  # outputs binary: negative/positive
    "nlptown": "nlptown/bert-base-multilingual-uncased-sentiment",  # outputs stars 1-5
    "bertweet": "finiteautomata/bertweet-base-sentiment-analysis"    # typically outputs binary sentiment
}

loaded_models = {}
for key, model_name in MODELS.items():
    tokenizer, model = load_model(model_name)
    loaded_models[key] = {"tokenizer": tokenizer, "model": model}

# Label mapping for the Cardiff model (3-class)
LABELS_roberta = ["negative", "neutral", "positive"]

def map_nlptown_rating(rating):
    # rating is expected to be an integer from 1 to 5
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:  # 4 or 5
        return "positive"

def analyze_sentiment_multi(text):
    """
    Run sentiment analysis on the input text using multiple models.
    Returns a dictionary with each model's sentiment label and probability scores.
    """

    # Sanitize input
    if isinstance(text, list):
        text = [t for t in text if isinstance(t, str) and t.strip()]
        if not text:
            return {"error": "All posts are empty or invalid."}
    elif isinstance(text, str):
        if not text.strip():
            return {"error": "Empty input text."}
        text = [text]  # convert single string to list for batching
    else:
        return {"error": "Invalid input format."}

    results = {}

    for key, resources in loaded_models.items():
        tokenizer = resources["tokenizer"]
        model = resources["model"]

        # Set max length based on model
        max_len = 128 if key == "bertweet" else 512

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        )

        # Extra safeguard against overflow
        if inputs['input_ids'].shape[1] > max_len:
            inputs['input_ids'] = inputs['input_ids'][:, :max_len]
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_len]

        with torch.no_grad():
            outputs = model(**inputs)

        probs = softmax(outputs.logits.numpy()[0])

        # Label handling per model
        if key == "cardiff":
            label = LABELS_roberta[probs.argmax()]
        elif key == "distilbert" or key == "bertweet":
            label = "positive" if probs[1] > probs[0] else "negative"
        elif key == "nlptown":
            star_rating = probs.argmax() + 1
            label = map_nlptown_rating(star_rating)
        else:
            label = "N/A"

        results[key] = {"label": label, "probs": probs.tolist()}

    return results