from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging
from langchain.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load the trained model and preprocessing objects
    model = joblib.load('models/reddit_post_popularity_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    pca = joblib.load('models/pca_transformer.pkl')
    selector = joblib.load('models/selected_features.pkl')
    logger.info("Models loaded successfully.")

except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e



# Define feature columns (excluding embeddings)
feature_cols = [
    'num_comments', 'upvote_ratio', 'text_length', 'hour', 'day', 'month', 'year', 'is_weekend',
    'sentiment_NEGATIVE', 'sentiment_POSITIVE',
    'subreddit_movies', 'subreddit_news', 'subreddit_politics',
    'subreddit_sports', 'subreddit_technology',
    'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
]

# Load the embeddings model
embeddings_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
try:
    embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    logger.info("Embeddings model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading embeddings model: {e}")
    raise e


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logger.info(f"Received data: {data}")

        # Extract text and generate embeddings
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'Text field is required.'}), 400

        text_embedding = embeddings_model.embed_documents([text])[0]
        embedding_df = pd.DataFrame([text_embedding], columns=[f'embedding_{i}' for i in range(len(text_embedding))])

        # Create DataFrame for input features
        input_df = pd.DataFrame([data])

        # Ensure all expected features are present
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Combine input features and embeddings
        input_combined = pd.concat([input_df[feature_cols], embedding_df], axis=1)

        # Apply scaling
        input_scaled = scaler.transform(input_combined)

        # Apply PCA to embeddings
        num_embeddings = len(text_embedding)
        embeddings_scaled = input_scaled[:, -num_embeddings:]
        embeddings_pca = pca.transform(embeddings_scaled)
        input_final = np.hstack((input_scaled[:, :-num_embeddings], embeddings_pca))

        # Create DataFrame for input features (including PCA features)
        input_final_df = pd.DataFrame(input_final, columns=feature_cols + [f'pca_{i}' for i in range(50)])

        # Filter to only the selected features
        input_filtered = input_final_df[selected_features]

        # Make prediction
        prediction = model.predict(input_filtered)
        logger.info(f"Prediction: {prediction[0]}")

        return jsonify({'predicted_score': float(prediction[0])})

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=2000)
