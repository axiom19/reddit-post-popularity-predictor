import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the PCA model, scaler, and trained model
pca = joblib.load('models/pca_transformer.pkl')        # Path to saved PCA model
scaler = joblib.load('models/scaler.pkl')        # Path to saved scaler
model = joblib.load('models/reddit_post_popularity_model.pkl')  # Path to trained regression model

# Load the sentence transformer model for embeddings
embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Sample input data
sample_data = {
    'title': "New technology for sports analytics",
    'selftext': "This is a post about the latest technology used in sports analytics.",
    'num_comments': 10,
    'upvote_ratio': 0.9,
    'is_weekend': 1,  # Example value
    'text_length': 150,  # Example text length
    'subreddit_movies': 0,
    'subreddit_news': 0,
    'subreddit_politics': 0,
    'subreddit_sports': 1,
    'subreddit_technology': 0,
    'sentiment_NEGATIVE': 0,
    'sentiment_POSITIVE': 1
}

# Step 1: Generate embeddings for the new input data
sample_text = sample_data['title'] + " " + sample_data['selftext']
embedding = embeddings_model.encode(sample_text).reshape(1, -1)  # Shape the embedding for one input

# Step 2: Apply PCA to reduce dimensionality of embeddings
reduced_embedding = pca.transform(embedding)

# Step 3: Combine PCA-transformed embeddings with additional features
additional_features = [
    sample_data['num_comments'],
    sample_data['upvote_ratio'],
    sample_data['is_weekend'],
    sample_data['text_length'],
    sample_data['subreddit_movies'],
    sample_data['subreddit_news'],
    sample_data['subreddit_politics'],
    sample_data['subreddit_sports'],
    sample_data['subreddit_technology'],
    sample_data['sentiment_NEGATIVE'],
    sample_data['sentiment_POSITIVE']
]

# Combine the PCA-reduced embeddings and additional features into a single array
combined_features = pd.DataFrame([additional_features + reduced_embedding.tolist()[0]])

# Step 4: Scale the combined features
scaled_features = scaler.transform(combined_features)

# Step 5: Use the trained model to make a prediction
prediction = model.predict(scaled_features)

# Output the result
print(f"Predicted score (upvotes) for the sample post: {prediction[0]}")