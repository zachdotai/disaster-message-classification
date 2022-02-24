# disaster-message-classification

### Context

Dataset containing real messages that were sent during disaster events was provided by [Appen](#https://www.appen.com). Categorizing incoming messages during a time of crisis can be of great help to emergency workers during a disaster.

This project includes a web app where an emergency worker can input a new message and get classification results (from a pre-trained multi-output classification model) in several categories. The purpose of which is to label disaster messages curated through social media, news outlets, and direct messages to a set of 36 labels to improve disaster response efficiency.

### Table of Contents

1. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Limitations](#results)
5. [Licensing](#licensing)

### Instructions  <a name="installation"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/message_categories.csv data/datasets.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/datasets.db models/classifier.pkl`

2. Go to `app` directory by running the following command in the terminal: `cd app`

3. Run the web app using the following command in the terminal: `python run.py`

4. Use the web page to query the model output for any specific message in mind!


### File Descriptions  <a name="files"></a>

```
.
├── README.md
├── app                                 -- Code for running the web app
│   ├── run.py                              -- Python web app run function
│   └── templates
│       ├── go.html                         -- Web page of app html
│       └── master.html                     -- Homepage of app html
├── data                                -- Input data and processed data
│   ├── datasets.db                     -- Preprocessed data
│   ├── disaster_categories.csv             -- Input data
│   ├── disaster_messages.csv               -- Input data
│   └── process_data.py                     -- Script to run data preprocessing
├── models                              -- Models
│   ├── classifier.pkl                      -- Saved model from training script
│   └── train_classifier.py                 -- script to rerun model fitting
└── notebooks                           -- Notebooks
    ├── Data Preprocessing.ipynb            -- Notebook used for data preprocessing
    └── Model Development.ipynb             -- Note used for preparing machine learning pipeline
```

### Limitations  <a name="results"></a>
Model accuracy was quite good after hyperparameter tuning, however, there are a number of limitations that could be part of future work of updating the model to improve its performance even further.

There is a significant imbalance in the quantity of training messages we have for each category, as can be seen from the web page prevalence of labels in the training dataset. If we use a naïve scoring function to fit the model, we can wind up with a model that does a good job at identifying "relevant" labels but fails miserably at classifying the more rare "fire" label.

Furthermore, it may be more necessary to decrease false-positives or false-negatives for particular categories. For other categories, capturing every event and putting up with a few false positives may be more crucial; for them, we should optimize for recall. If, on the other hand, the services that deal with a particular type of communication are overburdened with irrelevant messages (too many false-positives), we should focus on improving accuracy in that category.

Other improvements include trying out different tokenization function as well as including more features into the dataset. Genre and message length are some of the features that can be incorporated to improve model performance. This can perhaps be done by adding [Feature Union](#https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) into the ML training pipeline.

### Licensing  <a name="licensing"></a>
Please feel free to use the code here as you would like! 

