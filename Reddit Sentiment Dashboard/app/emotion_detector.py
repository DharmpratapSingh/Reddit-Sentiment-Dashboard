# app/emotion_detector.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Define the model to use for emotion detection
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Define the emotion labels (verify with the model card/documentation)
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def analyze_emotion(text):
    """
    Analyze the given text and return the predicted emotion and probability scores.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits.numpy()[0])
    emotion = LABELS[probs.argmax()]
    return emotion, probs.tolist()