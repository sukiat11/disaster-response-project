import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
pd.set_option('display.max_rows', None)

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath:str):
    '''
    Loads data from a database file

    Return X, Y dataframes and the category names
    '''
    engine = create_engine('sqlite:///%s'%database_filepath)
    df = pd.read_sql_table('messagesTable', engine)

    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, Y, Y.columns

def tokenize(text:str):
    '''
    Lematizes and converts text into tokens

    Returns a list of tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Initialize model pipeline, uses cross-validation to find the best parameters
    Model uses a Multi output classifier to allow for multi-target classification
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('message_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    ## For MultiOutputClassifier(LinearSVC())
    # parameters = {
    #     'features__message_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    #     'clf__estimator__C': [1, 2, 5]
    # }

    ## For MultiOutputClassifier(AdaBoostClassifier())
    parameters = {
        'features__message_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__learning_rate': [0.5, 0.75],
        'clf__estimator__n_estimators':[10, 50]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names:list):
    '''
    Returns metrics including precision, recall and f-1 score
    '''
    # Predict results
    Y_pred = pd.DataFrame(model.predict(X_test))
    model_type = "AdaBoostClassifier"
    title = '\n\nClassification Report using Multiouput %s'%model_type

    # Use heatmap to visualize report, only available with python 3.7 or later
    # If not available, results will still be displayed on terminal
    try:
        import seaborn as sns
        import matplotlib.pylab as plt

        class_report_results = classification_report(Y_test, Y_pred, target_names=category_names, output_dict=True)
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        class_report_df = pd.DataFrame(class_report_results).iloc[:-1, :].T
        ax = sns.heatmap(class_report_df, annot=True)
        ax.set_title(title)
        # plt.show()
        plt.savefig("./images/report_%.png"%model_type, bbox_inches='tight')
    except:
        class_report_df = classification_report(Y_test, Y_pred, target_names=category_names)
    print(f"\n\n{title}")
    print(class_report_df)


def save_model(model, model_filepath:str):
    '''
    Saves trained model to a pickle file
    '''
    try:
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)
        print('Trained model saved!')
    except:
        print("Error saving trained model.")

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

        print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()