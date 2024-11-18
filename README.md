# Amazon Fine Food Reviews Analysis

## Project Overview
This project focuses on analyzing the **Amazon Fine Food Reviews** dataset to derive meaningful insights and build predictive models. The dataset contains reviews from customers about various food products sold on Amazon India, enabling an in-depth exploration of customer sentiment and behavior.

## Objective
- Perform sentiment analysis to classify reviews as positive or negative.
- Explore and visualize key trends in the dataset.
- Develop predictive models to analyze customer sentiment based on review text.

---

## Dataset
- **Name**: Amazon Fine Food Reviews
- **Source**: Kaggle ([Link](https://www.kaggle.com/snap/amazon-fine-food-reviews))
- **Description**:
  - Contains over 500,000 reviews of fine foods on Amazon.
  - Includes attributes like:
    - Review Text
    - Ratings
    - Review Time
    - Helpful Scores
    - Summary
- **Purpose**: Analyze textual and numerical data to understand customer sentiments.

---

## Technologies and Tools
- **Programming Language**: Python
- **Libraries Used**:
  - **Data Processing**: Pandas, NumPy
  - **Visualization**: Matplotlib, Seaborn, Plotly
  - **NLP**: NLTK, spaCy, TF-IDF, Word2Vec
  - **Machine Learning**: Scikit-Learn, TensorFlow, PyTorch
  - **Deployment**: Flask (if applicable)
- **Development Environment**: Jupyter Notebook

---

## Steps in the Analysis
1. **Data Loading and Preprocessing**:
   - Clean and preprocess review text (e.g., removing stopwords, stemming, and lemmatization).
   - Handle missing values and perform exploratory data analysis.

2. **Exploratory Data Analysis (EDA)**:
   - Distribution of ratings and sentiment scores.
   - Trends in review length and helpful votes.
   - Visualize common keywords using word clouds.

3. **Feature Engineering**:
   - Extract features using **TF-IDF**, **Bag-of-Words**, and **Word2Vec**.
   - Create additional features based on helpful scores and review length.

4. **Model Development**:
   - Sentiment Classification using:
     - Logistic Regression
     - Support Vector Machines (SVM)
     - Random Forest
     - Deep Learning models (LSTM, BERT)
   - Compare model performance using metrics like accuracy, precision, recall, and F1-score.

5. **Evaluation**:
   - Evaluate models on a test set to ensure generalizability.
   - Visualize confusion matrices and ROC curves.

6. **Deployment** (Optional):
   - Create a REST API using Flask for predicting sentiment in real-time.

---

## Key Findings
- Positive reviews dominate the dataset, indicating customer satisfaction.
- Common keywords in negative reviews reveal critical feedback for improvement.
- LSTM models outperform traditional ML models for sentiment prediction.

