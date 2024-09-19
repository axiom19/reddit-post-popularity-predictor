# **Reddit Post Popularity Predictor**

Predict the popularity of Reddit posts using machine learning models. This project collects Reddit data, preprocesses it using advanced NLP techniques (including LangChain and Hugging Face transformers), performs exploratory data analysis, builds predictive models, and deploys a Flask API for real-time predictions.

## **Table of Contents**

- [Project Overview](#project-overview)
- [Features](#features)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [API Deployment](#api-deployment)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [Contact](#contact)

## **Project Overview**

This project aims to predict the popularity (upvote score) of Reddit posts before they are posted. By analyzing historical data and leveraging natural language processing, the model provides insights into how various factors contribute to a post's success.

## **Features**

- **Reddit Data Collection:** Fetch posts from specified subreddits using the Reddit API.
- **Advanced NLP Preprocessing:** Utilize LangChain and Hugging Face transformers for text embedding and sentiment analysis.
- **Exploratory Data Analysis:** Visualize data distributions and relationships between features.
- **Machine Learning Models:** Implement regression models including Random Forest, XGBoost, and Neural Networks.
- **Model Interpretation:** Use SHAP to interpret model predictions.
- **Flask API Deployment:** Provide an API endpoint for real-time predictions.

## **Data Collection**

Data is collected from the following subreddits:

- r/technology
- r/news
- r/movies
- r/sports
- r/politics

**Script:** `reddit_data_collection.py`

**Usage:**

1. Set up Reddit API credentials.
2. Run the data collection script to fetch posts.

## **Data Preprocessing**

The preprocessing pipeline includes:

- Handling missing values.
- Combining post title and content.
- Generating text embeddings using LangChain and Hugging Face models.
- Feature engineering (e.g., extracting datetime features, sentiment analysis).
- Encoding categorical variables.
- Scaling and normalizing numerical features.

**Script:** `data_preprocessing.py`

## **Exploratory Data Analysis**

EDA is performed to understand the data better:

- Visualize distributions of key variables.
- Analyze correlations and relationships.
- Identify and handle outliers.
- Document insights to inform model building.

**Notebook:** `exploratory_data_analysis.ipynb`

## **Model Building**

Multiple models are trained and evaluated:

- **Models Used:**
  - Random Forest Regressor
  - XGBoost Regressor
  - Neural Networks (MLP Regressor)

- **Model Interpretation:**
  - SHAP values are used to interpret feature importance and model predictions.

**Scripts:**

- `model_training.py`
- `model_evaluation.py`
- `model_interpretation.py`

## **API Deployment**

A Flask API is created to serve the model:

- **Endpoint:** `/predict`
- **Method:** POST
- **Input:** JSON containing post details.
- **Output:** JSON with the predicted score.

**Script:** `app.py`

**Deployment Options:**

- Local deployment for testing.
- Instructions provided for deploying to Heroku or other cloud services.

## **Getting Started**

### **Prerequisites**

- Python 3.7 or higher
- Reddit API credentials
- Git

### **Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/reddit-post-popularity-predictor.git
   cd reddit-post-popularity-predictor
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   - Create a `.env` file in the project root.
   - Add your Reddit API credentials and Hugging Face API token if necessary.

     ```env
     CLIENT_ID=your_reddit_client_id
     CLIENT_SECRET=your_reddit_client_secret
     USER_AGENT=your_app_user_agent
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
     ```

### **Usage**

1. **Data Collection:**

   ```bash
   python reddit_data_collection.py
   ```

2. **Data Preprocessing:**

   ```bash
   python data_preprocessing.py
   ```

3. **Model Training:**

   ```bash
   python model_training.py
   ```

4. **Start the API:**

   ```bash
   python app.py
   ```

5. **Make a Prediction:**

   Send a POST request to `http://localhost:5000/predict` with the required JSON data.

## **Repository Structure**

```
reddit-post-popularity-predictor/
├── app.py
├── data/
│   ├── reddit_posts.csv
│   └── reddit_posts_preprocessed.pkl
├── models/
│   ├── reddit_popularity_model.pkl
│   ├── scaler.pkl
│   └── pca_transformer.pkl
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── scripts/
│   ├── reddit_data_collection.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_interpretation.py
├── requirements.txt
├── README.md
└── .gitignore
```

## **Contributing**

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## **Contact**

**Your Name**

- **Email:** [:](shagundeepsingh80@gmail.com)
- **LinkedIn:** [My LinkedIn Profile](https://www.linkedin.com/in/shagundeep)
- **Portfolio:** [Portfolio](https://shagunnsingh007.wixsite.com/my-site)

---

## **How to Use This Project**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/reddit-post-popularity-predictor.git
   ```

2. **Follow the Installation Steps Above.**

3. **Run the Application:**

   - Start the Flask API.
   - Use a tool like Postman or `curl` to send test requests.

4. **Explore the Notebooks:**

   - Open the Jupyter notebooks in the `notebooks/` directory to see the EDA and model interpretation.

---

## **Feedback**

If you have any feedback or questions, please feel free to reach out or open an issue on GitHub.

---

## **Screenshots**

Included screenshots of:

### Demo Screenshots

- **Initial Data Visualization**  
  ![Data Visualization](https://github.com/axiom19/reddit-post-popularity-predictor/blob/main/demo/1.png)

- **Feature Importance Plot**  
  ![Feature Importance](https://github.com/axiom19/reddit-post-popularity-predictor/blob/main/demo/2.png)

- **Model Performance Plot**  
  ![Model Performance](https://github.com/axiom19/reddit-post-popularity-predictor/blob/main/demo/3.png)

- **SHAP Summary Plot**  
  ![SHAP Summary Plot](https://github.com/axiom19/reddit-post-popularity-predictor/blob/main/demo/4.png)

### Demo Test Video
  ![Demo Video](https://github.com/axiom19/reddit-post-popularity-predictor/blob/main/demo/demo.mov)
