import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads the data from the locally saved files
    Args:
        messages_filepath (str) - Path to messages.csv file
        categories_filepath (str) - Path to messages.csv file
    Returns:
        loaded_dataset (pd.DataFrame) - The merged dataset (unprocessed) of the two files
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    loaded_dataset = messages.merge(categories, left_on='id', right_on='id')
    
    return loaded_dataset

def clean_data(df):
    '''
    Cleans the data by processing previously identified issues with the original dataset
    Args:
        df (pd.DataFrame) - DataFrame object of the unprocessed dataset
    Returns:
        preprocessed_dataset (pd.DataFrame) - The cleaned and preprocessed dataset for model training
    '''
    
    split_categories = df['categories'].str.split(pat=';', expand=True)

    first_row = split_categories.head(1).squeeze()

    # use this row to extract a list of new column names for categories.
    category_column_names = first_row.str.split(pat='-', n=1, expand=True)[0]
        
    split_categories.columns = category_column_names
    
    for column in split_categories:
        # set each value to be the last character of the string
        split_categories[column] = split_categories[column].str.split(pat='-', n=1, expand=True)[1]

        # convert column from string to numeric
        split_categories[column] = split_categories[column].astype(int)
    
    # Replace values of column 'related' that are 2 with 1
    split_categories['related'] = np.where(split_categories['related']!=0,1,0)
    
    # Remove all classes where only one label exists in the dataset since no information
    # can be inferred about this class and it causes some algorithms to break
    split_categories = split_categories.loc[:,split_categories.describe().loc['max']!=0]
    
    preprocessed_dataset = df.drop('categories', axis=1)
    
    preprocessed_dataset = pd.concat([preprocessed_dataset, split_categories], axis=1)
    
    # Check and remove duplicates rows
    preprocessed_dataset = preprocessed_dataset.drop_duplicates()
    
    return preprocessed_dataset

    
def save_data(df, database_filename):
    '''
    Saved the preprocessed dataset into the local database
    Args:
        df (pd.DataFrame) - DataFrame object of the preprocessed dataset
    Returns:
        None
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('labelled_messages', engine, index=False, if_exists='replace')


def main():
    '''
    Main function to preprocess dataset for model training. It takes three arguments as outlined below is the respective order
    Args:
        messages_filepath (str) - Path to load unprocessed messages.csv file
        categories_filepath (str) - Path to load unprocessed messages.csv file
        database_filepath )str) - Path to save the preprocessed data into database file
    Returns:
        None
    '''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
