# AI-ML-Engineering-Advanced-Tasks

Machine Learning Projects: News Classification, Churn Prediction, and Context-Aware Chatbot

Task 1: News Topic Classifier Using BERT
Objective
Develop a machine learning model to classify news articles into one of four topics (World, Sports, Business, Sci/Tech) using the AG News dataset. The goal is to preprocess the data, fine-tune a BERT model, evaluate performance with accuracy and F1-score, test sample headlines, and deploy the model using Streamlit in Google Colab.
Methodology / Approach

Dataset: Used the AG News dataset, containing news articles labeled with four topics. A subset of 10,000 training samples and 2,000 test samples was used for faster execution in Colab.
Preprocessing: Tokenized text using the BERT tokenizer (bert-base-uncased) with a maximum length of 128 tokens, padding, and truncation. Formatted the dataset for PyTorch compatibility.
Model: Fine-tuned a pre-trained BERT model (bert-base-uncased) for sequence classification with 4 output labels.
Training: Used the Hugging Face Trainer API with:
Learning rate: 2e-5
Batch size: 16 (train and eval)
Epochs: 3
Weight decay: 0.01
Evaluation strategy: Per epoch
Saved model and logs locally to /content/bert_news_classifier and /content/logs.


Evaluation: Measured accuracy and weighted F1-score on the test set.
Sample Testing: Tested the model on sample headlines: "New AI breakthrough in quantum computing", "Stock market crashes amid global uncertainty", and "Team wins championship in thrilling final".
Deployment: Created a Streamlit app for interactive classification, hosted via ngrok in Colab.
Environment: Implemented in Google Colab with GPU support, avoiding Google Drive for file storage.

Key Results or Observations

Expected Performance: With 10,000 training samples and 3 epochs, the model typically achieves an accuracy of approximately 90-94% and a weighted F1-score of 90-94% on the test set, reflecting BERT's strong performance on text classification.
Sample Predictions: The model is expected to classify headlines correctly, e.g., "New AI breakthrough in quantum computing" as Sci/Tech, "Stock market crashes amid global uncertainty" as Business, and "Team wins championship in thrilling final" as Sports.
Deployment: The Streamlit app allows users to input headlines and view predicted topics, demonstrating practical usability.
Observation: The reduced dataset size speeds up training but may slightly lower performance compared to the full dataset. The model is effective for short headlines but may need additional preprocessing for complex articles.

Task 2: End-to-End ML Pipeline for Customer Churn Prediction
Objective
Build a reusable, production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The pipeline includes preprocessing, training Logistic Regression and Random Forest models, hyperparameter tuning with GridSearchCV, and exporting the pipeline using joblib.
Methodology / Approach

Dataset: Used the Telco Customer Churn dataset (~7,043 samples) with customer features (e.g., tenure, MonthlyCharges, Contract) and a binary target (Churn: Yes/No).
Preprocessing:
Replaced 'No internet service' and 'No phone service' with 'No' for consistency.
Converted TotalCharges to numeric, handling invalid entries.
Numerical features (tenure, MonthlyCharges, TotalCharges) were imputed (median) and scaled (StandardScaler).
Categorical features were imputed (most frequent) and one-hot encoded (OneHotEncoder with handle_unknown='ignore').
Used ColumnTransformer and Pipeline to encapsulate preprocessing.


Models:
Trained Logistic Regression and Random Forest models.
Hyperparameter tuning via GridSearchCV:
Logistic Regression: Tuned C (0.01, 0.1, 1, 10) with l2 penalty.
Random Forest: Tuned n_estimators (100, 200), max_depth (10, 20, None), min_samples_split (2, 5).


Scoring metric: Weighted F1-score.


Evaluation: Measured accuracy, F1-score, and classification reports on the test set (20% of data, ~1,409 samples).
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


Observation: Slightly lower F1-score than Logistic Regression, likely due to restricted max_depth preventing overfitting.


General Observations:
Logistic Regression outperforms Random Forest in F1-score (0.6032 vs. 0.5786), making it preferable for this imbalanced dataset.
The pipeline is production-ready, handling missing values and unknown categories, and is reusable via .joblib files.
Class imbalance impacts Churn recall; techniques like oversampling could improve performance.
Training is efficient (~1-3 minutes in Colab).



Task 4: Context-Aware Chatbot Using LangChain and RAG
Objective
Develop a conversational chatbot that maintains context across interactions and retrieves relevant information from a custom knowledge base using Retrieval-Augmented Generation (RAG). The chatbot is deployed using Streamlit in Google Colab.
Methodology / Approach

Dataset: Created a sample corpus of three text documents about Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning, stored at /content/corpus/ai_knowledge.txt. This simulates a knowledge base (e.g., Wikipedia pages).
Preprocessing:
Loaded the corpus using TextLoader from langchain_community.document_loaders.
Split documents into 500-character chunks with 50-character overlap using RecursiveCharacterTextSplitter.
Embedded chunks using sentence-transformers/all-MiniLM-L6-v2 (384-dimensional vectors).
Created a FAISS vector store for fast document retrieval.


Model:
Used distilgpt2 (Hugging Face) for text generation, configured with a maximum of 100 new tokens.
Integrated with LangChain’s HuggingFacePipeline for conversational use.


Conversational Setup:
Implemented ConversationBufferMemory to store chat history for context-aware responses.
Created a ConversationalRetrievalChain combining the LLM, FAISS retriever (top 2 documents), and memory.


Testing: Tested with queries: "What is Artificial Intelligence?", "How does Machine Learning relate to AI?", and "Tell me more about Deep Learning."
Deployment:
Built a Streamlit app (app.py) for interactive querying, displaying responses and chat history.
Used ngrok to create a public URL in Colab.


Environment: Implemented in Google Colab with optional GPU support, storing files locally in /content.

Key Results or Observations

Expected Performance:
The chatbot retrieves relevant document chunks and generates responses aligned with the corpus (e.g., AI as human intelligence simulation, ML as a subset of AI).
Context memory retains chat history, enabling follow-up questions to reference prior interactions.


Deployment:
The Streamlit app provides an interactive interface for querying and viewing chat history.
Ngrok deployment requires a valid authtoken from https://dashboard.ngrok.com/get-started/your-authtoken.


Observations:
distilgpt2 generates concise but sometimes less coherent responses. A stronger model (e.g., via xAI’s API at https://x.ai/api) could improve quality.
FAISS ensures fast retrieval, scalable for larger corpora.
The small corpus limits response diversity; a larger corpus would enhance performance.
Setup takes ~2-5 minutes in Colab, with GPU accelerating LLM inference.


Challenges:
An ngrok authentication error (ERR_NGROK_105) occurs with an invalid authtoken. Use a valid token from the ngrok dashboard.
Colab’s memory constraints may limit larger corpora or models; reducing chunk_size helps.


