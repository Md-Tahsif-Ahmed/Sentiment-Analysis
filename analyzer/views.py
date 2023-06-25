import os
import torch
from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from .serializers import SentimentAnalysisSerializer


model_path = "Sentiment_API/sentiment_model"

if not os.path.exists(model_path):
    model_name = "StatsGary/setfit-ft-sentinent-eval"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path)




@api_view(['GET', 'POST'])
def analyze(request):
    if request.method == 'POST':
        serializer = SentimentAnalysisSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            sentiment = perform_sentiment_analysis(text)
            response_data = {'sentiment': sentiment}
            return Response(response_data)
        else:
            return Response(serializer.errors, status=400)
    elif request.method == 'GET':
        # Handle GET request logic here
        # For example, you can return a form or instructions
        return Response({'message': 'Please use POST request to analyze text'})
def perform_sentiment_analysis(text):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize the text
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Perform sentiment analysis
    outputs = model(**encoded_input)
    predicted_class = outputs.logits.argmax(dim=1).item()

    sentiment = get_sentiment_label(predicted_class)
    return sentiment


def get_sentiment_label(sentiment_id):
    # Map sentiment IDs to labels (modify according to your sentiment labels)
    sentiment_labels = {
        0: 'negative',
        1: 'positive',
        2: 'neutral'
        
    }
    return sentiment_labels.get(sentiment_id, 'unknown')
