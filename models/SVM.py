import os
import pickle

from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import  LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
#from sklearn.metrics import classification_report
from sklearn.metrics import r2_score, mean_squared_error

import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import nltk

nltk.download('stopwords')


def create_train_model(train, validate, test, attribute):
    print(f"Creating model for {attribute}")

    print("Reading data into dataframe")
    df = pd.read_csv(train)
    #df = df[df['stars'] != 1]
    #df = df[df['stars'] != 2]

    #df_binary = df[df['stars'].isin([1, 2, 4, 5])]
    #df_binary['label'] = (df_binary['stars'] > 3).astype(int) 
    #attribute = 'label'

    df_sampled = df.sample(n=15000, random_state=42)
    X = df_sampled['text']
    y = df_sampled[attribute]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline
    print("Creating pipeline")
    pipeline = {}
    param_grid = {}

    if (attribute == 'stars'):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('svm', SVC())
        ])
    # Define the parameter grid
        print("Defining parameter grid")
        param_grid = {
            'tfidf__max_features': [2000],
            'tfidf__ngram_range': [(1, 2)],
            'svm__C': [10],
            'svm__kernel': ['rbf']
        }
    else:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('svr', SVR())
        ])

# Define the parameter grid
        param_grid = {
            'tfidf__max_features': [2000],
            'tfidf__ngram_range': [(1, 2)],
            'svr__C': [10],
            'svr__kernel': ['rbf'],
        }

    # Setup the grid search
    print("Setting up grid search")

    grid_search = {}
    if (attribute == 'stars'):
        grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1, scoring='f1_macro', cv=5)
    else:
        grid_search = GridSearchCV(pipeline, param_grid, n_jobs=-1, scoring='r2', cv=5)
    
    
    # Fit the grid search to the data
    print("Fitting grid search")
    grid_search.fit(X_train, y_train)

    # Best parameter set found on development set
    print("Best parameters found:\n", grid_search.best_params_)

    # Evaluate the best model found by the grid search on the test set
    best_model = grid_search.best_estimator_
    filename = f'SVM_{attribute}.pkl'
    pickle.dump(best_model, open(filename, 'wb'))
    print(f"Model saved as {filename}")


    predictions = best_model.predict(X_test)
    print("Predicted values...")
    if (attribute == 'stars'):
        print("Classification report:\n", classification_report(y_test, predictions))
    else:
        print("R2 score:\n", r2_score(y_test, predictions))

def use_train_model(test_data, target):
    filename = f'SVM_{target}.pkl'
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
        print(f"Model loaded from {filename}")

        print("Reading data into dataframe")
        df = pd.read_csv(test_data)
        #df = df[df['stars'] != 1]
        #df = df[df['stars'] != 2]
        df_sampled = df.sample(n=40000, random_state=42)
        X = df_sampled['text']
        y = df_sampled[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        predictions = loaded_model.predict(X_test)

        if (target == 'stars'):
            print("Classification report:\n", classification_report(y_test, predictions))
        else:
            print("R2 score:\n", r2_score(y_test, predictions))


    