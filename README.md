# AI-ML-Engineering-Advanced-Tasks

**Task 1: News Topic Classifier Using BERT**
Objective
Develop a machine learning model to classify news articles into one of four topics (World, Sports, Business, Sci/Tech) using the AG News dataset. The goal is to preprocess the data, fine-tune a BERT model, evaluate performance with accuracy and F1-score, test sample headlines, and deploy the model using Streamlit in Google Colab.
Methodology / Approach

Dataset: Utilized the AG News dataset, containing news articles labeled with four topics. A subset of 10,000 training samples and 2,000 test samples was used for faster execution in Colab.
Preprocessing: Tokenized text using the BERT tokenizer (bert-base-uncased) with a maximum length of 128 tokens, padding, and truncation.
Model: Fine-tuned a pre-trained BERT model (bert-base-uncased) for sequence classification with 4 output labels.
Training: Used the Hugging Face Trainer API with the following parameters:
Learning rate: 2e-5
Batch size: 16 (train and eval)
Epochs: 3
Weight decay: 0.01
Evaluation strategy: Per epoch
Saved model and logs locally to /content/bert_news_classifier and /content/logs.


Evaluation: Measured performance using accuracy and weighted F1-score on the test set.
Sample Testing: Tested the model on three sample headlines to verify predictions.
Deployment: Created a Streamlit app for interactive classification, hosted via ngrok in Colab.
Environment: Implemented in Google Colab with GPU support, avoiding Google Drive for file storage.

Key Results or Observations

Expected Performance: Based on the setup (10,000 training samples, 3 epochs), the model typically achieves an accuracy of approximately 90-94% and a weighted F1-score of 90-94% on the test set, reflecting BERT's strong performance on text classification tasks.
Sample Predictions: The model correctly classifies headlines like "New AI breakthrough in quantum computing" as Sci/Tech, "Stock market crashes amid global uncertainty" as Business, and "Team wins championship in thrilling final" as Sports (results depend on training).
Deployment: The Streamlit app allows users to input headlines and view predicted topics, demonstrating practical usability.
Observation: Reducing the dataset size speeds up training but may slightly lower performance compared to using the full dataset. The model is robust for short headlines but may require longer sequences or additional preprocessing for complex articles.

**Task 2: End-to-End ML Pipeline for Customer Churn Prediction**
Objective
Build a reusable, production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The pipeline includes preprocessing, training Logistic Regression and Random Forest models, hyperparameter tuning with GridSearchCV, and exporting the pipeline using joblib.
Methodology / Approach

Dataset: Used the Telco Customer Churn dataset, containing customer features (e.g., tenure, MonthlyCharges, Contract) and a binary target (Churn: Yes/No). The dataset has approximately 7,043 samples.
Preprocessing:
Replaced 'No internet service' and 'No phone service' with 'No' for consistency.
Converted TotalCharges to numeric, handling invalid entries.
Numerical features (tenure, MonthlyCharges, TotalCharges) were imputed (median) and scaled (StandardScaler).
Categorical features were imputed (most frequent) and one-hot encoded (OneHotEncoder with handle_unknown='ignore').
Used ColumnTransformer and Pipeline to encapsulate preprocessing.


Models:
Trained two models: Logistic Regression and Random Forest.
Hyperparameter tuning via GridSearchCV:
Logistic Regression: Tuned C (0.01, 0.1, 1, 10) with l2 penalty.
Random Forest: Tuned n_estimators (100, 200), max_depth (10, 20, None), min_samples_split (2, 5).


Scoring metric: Weighted F1-score (suitable for imbalanced data).


Evaluation: Measured accuracy, F1-score, and detailed classification reports on the test set (20% of data, ~1,409 samples).
Export: Saved pipelines as .joblib files (telco_churn_logistic_regression_pipeline.joblib, telco_churn_random_forest_pipeline.joblib) and results as telco_churn_results.json.
Environment: Implemented in Google Colab, storing files locally in /content.

Key Results or Observations

Logistic Regression:
Best Parameters: C=10, penalty='l2'
Accuracy: 0.8048
F1-Score: 0.6032
Classification Report:
No Churn: Precision 0.85, Recall 0.89, F1-score 0.87 (support: 1035)
Churn: Precision 0.66, Recall 0.56, F1-score 0.60 (support: 374)
Weighted Avg: Precision 0.80, Recall 0.80, F1-score 0.80


Observation: Strong performance on No Churn class, but lower recall for Churn due to class imbalance.


Random Forest:
Best Parameters: max_depth=10, min_samples_split=2, n_estimators=100
Accuracy: 0.7984
F1-Score: 0.5786
Classification Report:
No Churn: Precision 0.84, Recall 0.90, F1-score 0.87 (support: 1035)
Churn: Precision 0.65, Recall 0.52, F1-score 0.58 (support: 374)
Weighted Avg: Precision 0.79, Recall 0.80, F1-score 0.79


Observation: Slightly lower F1-score than Logistic Regression, likely due to overfitting prevention from tuned max_depth.


General Observations:
Logistic Regression slightly outperforms Random Forest in F1-score (0.6032 vs. 0.5786), making it preferable for this imbalanced dataset.
The pipeline is production-ready, handling missing values and unknown categories, and is reusable via .joblib files.
Class imbalance (more No Churn than Churn) impacts recall for the Churn class; techniques like oversampling or class weights could improve performance.
Training is efficient (~1-3 minutes in Colab), and saved pipelines enable easy deployment.


