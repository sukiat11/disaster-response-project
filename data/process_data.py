import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df

def clean_data(df):
    categories = df['categories'].str.split(";", expand=True)

    row = categories.iloc[0].tolist()
    category_colnames = [s[:-2] for s in row]
    categories.columns = category_colnames

    # Convert category values to just numbers 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        # Turn 2's into 1's to ensure binary format
        categories[column] = categories[column].apply(lambda x: 1 if x>0 else 0)

    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    df = df.reset_index(drop=True)

    # Remove extra characters and multiple spaces from col
    # Normalize to uppercase
    df['message'] = df['message'].str.replace(r"[^a-zA-Z0-9]", " ", regex=True).str.lower()
    df['message'] = df['message'].str.replace(r"\s\s+", " ", regex=True).str.strip()

    # Remove column that has all values the same. Not useful for training
    df = df.drop(['child_alone'], axis=1)

    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///%s'%database_filename)
    df.to_sql('messagesTable', engine, index=False, if_exists='replace')


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
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
