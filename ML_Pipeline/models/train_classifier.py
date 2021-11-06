import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import type_of_target
nltk.download(['punkt', 'wordnet'])
import os.path
import time


def load_data(database_filepath):
    '''
    :param database_filepath: file path of database file
    :return:
    X: Training model input , messages in this instance
    y: Dependent variable
    category_names: list with names of the target variables
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM data", engine)
    X = df.message

    category_names = list(df.iloc[:, 4:].columns)
    y = df[category_names]
    return X,y, category_names

def tokenize(text):
    '''

    :param text: String that you want tokenize
    :return: Clean tokens (lowercase, without punctuation, lemmatized)
    '''
    # convert to lowercase
    text = text.lower()
    # remove punctuation
    # Remove punctuation characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    # tokenize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    :return: Model pipeline using Grid Search
    '''
    # model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Create Grid search parameters
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 70]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=4)


    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """Evalute metrics of the ML pipeline model

    inputs:
    npTrue: array. Array containing the real labels.
    npPred: array. Array containing predicted labels.
    category_names: list of strings. List containing names for each of the ArrayP fields.

    Returns:
    data_metrics: Contains accuracy, precision, recall
    and f1 score for a given set of ArrayL and ArrayP labels.
    """

    # predict
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[category_names[i]], y_pred[:, i]),zero_division=0)

    # npTrue = np.array(Y_test)
    # npPred = np.array(y_pred)
    # metrics = []
    #
    # # Evaluate metrics for each set of labels
    # for i in range(len(category_names)):
    #     accuracy = accuracy_score(npTrue[:, i], npPred[:, i])
    #     precision = precision_score(npTrue[:, i], npPred[:, i], average="weighted",zero_division=0)
    #     recall = recall_score(npTrue[:, i], npPred[:, i], average="weighted")
    #     f1 = f1_score(npTrue[:, i], npPred[:, i], average="weighted")
    #
    #     metrics.append([accuracy, precision, recall, f1])
    #
    # # store metrics
    # metrics = np.array(metrics)
    # data_metrics = pd.DataFrame(data=metrics, index=category_names, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
    #
    # print(data_metrics.describe())
    # return data_metrics

def save_model(model, model_filepath):
    '''

    :param model: Model instance that we want to save
    :param model_filepath: string with filepath where model exists
    :return:
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()



        model.fit(X_train, Y_train)

        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    elif (os.path.exists(os.path.join("..","data","data.db"))):

        database_filepath = os.path.join("..","data","data.db")
        model_filepath = "classifier.pkl"
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        tic = time.perf_counter()
        model.fit(X_train, Y_train)
        toc = time.perf_counter()
        print(f"Time: {(toc - tic) / 60:0.4f} minutes")

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')



    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/data.db classifier.pkl')


if __name__ == '__main__':
    main()