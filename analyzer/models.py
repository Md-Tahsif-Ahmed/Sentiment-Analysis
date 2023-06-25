from django.db import models

class Sentiment(models.Model):
    text = models.TextField()

    class Meta:
        app_label = 'analyzer'

    def __str__(self):
        return self.text

 