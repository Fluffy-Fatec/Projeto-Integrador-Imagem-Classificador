from repository import SentimentRepository

class SentimentService:
    @staticmethod
    def save_sentiment(sentiment):
        return SentimentRepository.save_sentiment(sentiment)