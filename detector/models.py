from django.db import models

class NewsHistory(models.Model):
    news_text = models.TextField()
    prediction = models.CharField(max_length=50)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction} - {self.created_at}"
