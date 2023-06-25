from rest_framework import serializers
from .models import Sentiment

class SentimentAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sentiment
        fields = ['text']

