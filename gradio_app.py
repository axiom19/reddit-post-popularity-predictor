import gradio as gr
import requests

# Define the URL of your Flask API
API_URL = 'http://127.0.0.1:5000/predict'

def predict_popularity(title, selftext, num_comments, upvote_ratio, is_weekend, text_length, subreddit, sentiment):
    # Prepare the data for the API request
    data = {
        "title": title,
        "selftext": selftext,
        "num_comments": num_comments,
        "upvote_ratio": upvote_ratio,
        "is_weekend": is_weekend,
        "text_length": text_length,
        "subreddit_movies": 1 if subreddit == 'movies' else 0,
        "subreddit_news": 1 if subreddit == 'news' else 0,
        "subreddit_politics": 1 if subreddit == 'politics' else 0,
        "subreddit_sports": 1 if subreddit == 'sports' else 0,
        "subreddit_technology": 1 if subreddit == 'technology' else 0,
        "sentiment_NEGATIVE": 1 if sentiment == 'NEGATIVE' else 0,
        "sentiment_POSITIVE": 1 if sentiment == 'POSITIVE' else 0
    }

    # Send the POST request to the Flask API
    response = requests.post(API_URL, json=data)

    # Parse the response from the API
    if response.status_code == 200:
        return response.json()['predicted_score']
    else:
        return "Error: Could not get prediction"

# Create the Gradio interface with the new component API
title_input = gr.Textbox(label="Post Title")
selftext_input = gr.Textbox(label="Post Content")
num_comments_input = gr.Number(label="Number of Comments", value=0)
upvote_ratio_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Upvote Ratio")
is_weekend_input = gr.Radio(choices=[1, 0], label="Is it the Weekend?")
text_length_input = gr.Number(label="Text Length", value=0)
subreddit_input = gr.Dropdown(choices=['movies', 'news', 'politics', 'sports', 'technology'], label="Subreddit")
sentiment_input = gr.Dropdown(choices=['POSITIVE', 'NEGATIVE'], label="Sentiment")

# Set up the interface function and layout
interface = gr.Interface(
    fn=predict_popularity,
    inputs=[title_input, selftext_input, num_comments_input, upvote_ratio_input, is_weekend_input, text_length_input, subreddit_input, sentiment_input],
    outputs="text",
    title="Reddit Post Popularity Predictor",
    description="Predict the popularity score of a Reddit post based on its content and metadata."
)

interface.launch()
