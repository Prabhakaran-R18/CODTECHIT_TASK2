# CODTECHIT_TASK2

## Name: Prabhakaran R

## Domain: Machine Learning

## Intern ID : CT08ETY

## Company: CODTECHIT SOLUTIONS

# PROJECT : Movie Review Sentiment Analysis

## Project Overview
This project implements a sentiment analysis model that classifies movie reviews as positive or negative. Using Natural Language Processing (NLP) techniques and Machine Learning, the system analyzes the text content of movie reviews and predicts sentiment ratings.

## Dataset link : [Uploading kalki_movie_reviews.csv…]()

## Features
- Text preprocessing and vectorization using CountVectorizer
- Multinomial Naive Bayes classifier for sentiment prediction
- Interactive review rating prediction
- Support for processing movie reviews in text format
- Automatic rating classification

## Technologies Used
- Python 3.x
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn: Machine Learning tools
  - CountVectorizer for text feature extraction
  - MultinomialNB for classification
  - train_test_split for dataset splitting

- Ensure your CSV file contains 'Comments' and 'Ratings' columns
- Place the dataset in the project directory

## Usage
The model follows these steps:

1. Data Loading and Preprocessing:
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
ds = pd.read_csv('kalki_movie_reviews.csv')

# Split features and target
features = ['Comments']
target = 'Ratings'
x = ds[features]
y = ds[target]
```

2. Model Training:
```python
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train['Comments'])

# Train the model
model = MultinomialNB()
model.fit(x_train_vec, y_train)
```

3. Making Predictions:
```python
# Input processing
input_review = "Your movie review text here"
input_review_vec = vectorizer.transform([input_review])
prediction = model.predict(input_review_vec)
```

## Model Performance
The model's accuracy is evaluated using scikit-learn's accuracy_score metric. Current performance metrics:
- Training-Test Split: 75%-25%
- Model: Multinomial Naive Bayes
- Feature Extraction: Count Vectorization

## Future Improvements
1. Implement advanced text preprocessing techniques
2. Experiment with different ML algorithms (SVM, LSTM)
3. Add cross-validation
4. Enhance feature engineering
5. Include sentiment intensity analysis

PROJECT RESULTS :

![ttwo1](https://github.com/user-attachments/assets/8505407c-ff6f-44d2-a71e-b42162e437db)

![ttwo2](https://github.com/user-attachments/assets/30d08fb4-2277-4d5c-9226-5c29cd0a23ca)

![ttwo3](https://github.com/user-attachments/assets/de8b968c-39c3-4e45-ae60-53fe57810f89)

![ttwo4](https://github.com/user-attachments/assets/82220973-586f-481d-bb0c-187874866a20)

![ttwo5](https://github.com/user-attachments/assets/e309513f-f875-4f8b-b10d-3880f87fb82c)

![ttwo6](https://github.com/user-attachments/assets/41414140-880c-4d4d-ad42-ef2437a254d3)






