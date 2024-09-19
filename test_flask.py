import requests

API_URL = 'http://127.0.0.1:5000/predict'

data = {
    "title": "Test Title",
    "selftext": "Test content",
    "num_comments": 5,
    "upvote_ratio": 0.75,
    "is_weekend": 0,
    "text_length": 100,
    "subreddit_movies": 0,
    "subreddit_news": 1,
    "subreddit_politics": 0,
    "subreddit_sports": 0,
    "subreddit_technology": 0,
    "sentiment_NEGATIVE": 0,
    "sentiment_POSITIVE": 1
}

response = requests.post(API_URL, json=data)

print(response.status_code)
print(response.json())
