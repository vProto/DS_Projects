import sys
import pandas as pd
from sqlalchemy import create_engine
import os.path

def load_data(messages_filepath, categories_filepath):
    """Load data.

    inputs:
    messages_filepath: the path where the messages data exist
    categories_filepath: the path where the categories data exist

    outputs:
    df: dataframe with both datasets
    """
    # Load Messages Dataset
    messages = pd.read_csv(messages_filepath)

    # Load Categories Dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.merge(messages, categories, on="id")
    return df



def clean_data(df):
    """Clean data.

    inputs:
    df: Dataframe with messages and categories data

    outputs:
    df: Cleaned Dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df = df.drop(["categories"], axis=1)


    # Concatenate   dataframe with the new `categories`
    df = pd.concat([df, categories], axis = 1)
    # Drop duplicates
    df.drop_duplicates(inplace = True)


    # Remove rows with a  value of 2 from df
    df = df[df['related'] != 2]

    return df


def save_data(df, database_filename):
    """Save into  SQLite database.

    inputs:
    df: dataframe. Dataframe containing cleaned version of merged message and
    categories data.
    database_filename: string. Filename for output database.

    outputs:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('data', engine, index=False, if_exists='replace')


def main():
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

    elif (os.path.exists("disaster_categories.csv")
          & os.path.exists("disaster_messages.csv")
          & os.path.exists("data.db")):

        messages_filepath = "disaster_messages.csv"
        categories_filepath = "disaster_categories.csv"
        database_filepath = "data.db"

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