import json
import plotly
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    Tokenize text by removing stopwords, reducing words to their stems, and lemmatizing words to their root form
    Args:
        text(str) - Text to be tokenized ahead of mode training/prediction
    Returns:
        X (pd.Series) - The preprocessed dataset for the features
        Y (pd.DataFrame) - The preprocessed dataset for the target variables
        category_names (pd.Index) - The labels for messages to be categorized into
    '''
    
    tokens = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = text.split()
    
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    # Reduce words to their stems
    tokens = [PorterStemmer().stem(t) for t in tokens]
    
    # Reduce words to their root form
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    
    return tokens

# load data
engine = create_engine('sqlite:///../data/datasets.db')
df = pd.read_sql_table('labelled_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''Index web page handles user message and show model prediction'''

    # extract data needed for visuals
    labels = df.iloc[:,4:]
    
    label_prevalence = labels.mean().values
    label_names = list(labels.mean().index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_prevalence
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Labels',
                'yaxis': {
                    'title': "% Prevalence in Dataset"
                },
                'xaxis': {
                    'title': "Label"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''Web page that uses user query to predict disaster label of user message'''
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
