import sys

import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import re

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report

from sqlalchemy import create_engine

import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    preprocessed_dataset = pd.read_sql_table('labelled_messages', engine)
    
    X = preprocessed_dataset['message']
    Y = preprocessed_dataset.iloc[:,4:]
    
    category_names = Y.columns
    
    return X, Y, category_names
    
def tokenize(text):
    tokens = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = text.split()
    
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    # Reduce words to their stems
    tokens = [PorterStemmer().stem(t) for t in tokens]
    
    # Reduce words to their root form
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,1))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier(loss='modified_huber', alpha=0.0001)))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
        
    Y_pred = model.predict(X_test)
        
    accuracy = (Y_pred == Y_test).mean()

    print("Labels:", category_names)
    print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump( model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()