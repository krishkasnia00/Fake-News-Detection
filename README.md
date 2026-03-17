Problem Statement
Fake news is a type of news that is spread quickly in digital media and causes misinformation and harms society. The current systems are either computationally expensive (Deep Learning) or fail to understand the context (Traditional ML).
The goal of this project is to create a system that balances:
Accuracy
Efficiency
Scalability

📌 Abstract
With the rise of digital technology, fake news is disseminated at a rapid pace, affecting society, politics, and economics.
However, verification of such news is a tedious job due to the sheer volume of information.
This project aims to develop an automated system to detect fake news using:
- Natural Language Processing
- TF-IDF Feature Extraction
- Machine Learning Classifiers
The system can classify news with high accuracy and efficiency.

🎯 Objectives
To know what fake news is and its impact
To develop a classification system using ML algorithms
To apply NLP preprocessing and TF-IDF feature extraction
To evaluate the performance of the system using standard parameters
To develop an efficient system

🚀 Features
Automated fake vs real news classification
Text preprocessing using NLP
Feature extraction using TF-IDF
ML models: Logistic Regression, Naïve Bayes
Performance evaluation (Accuracy, Precision)

🛠️ Technologies Used
Programming Language
Python
Libraries & Tools
Scikit-learn
Pandas
NumPy
NLTK
Development Environment
Jupyter Notebook / VS Code

fake-news-detection/
│── data/
│   ├── dataset.csv
│── notebooks/
│   └── analysis.ipynb
│── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   └── predict.py
│── models/
│   └── trained_model.pkl
│── requirements.txt
│── README.md

⚙️ Methodology
The system follows a structured pipeline:
Data Collection
Datasets from Kaggle (ISOT, LIAR, Fake News datasets)
Data Preprocessing
Lowercasing
Stopword removal
Tokenization
Noise removal
Feature Extraction
TF-IDF converts text into numerical vectors
Model Training
Logistic Regression
Naïve Bayes
Model Evaluation
Accuracy
Precision
Prediction
Classifies unseen news as Fake or Real
