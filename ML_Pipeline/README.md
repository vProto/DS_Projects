# Messages Classification ML Pipeline


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Project Descriptions](#descriptions)
4. [Files Descriptions](#files)
5. [Instructions](#instructions)

## Installation <a name="installation"></a>

All libraries are available in Anaconda distribution of Python. The used libraries are:

- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

The code should run using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

This is a sample machine learning project that combines:
- ETL pipeline to extract, transform and load the data in a SQL database
- ML pipeline to train and tune the classifier that is used for predictions
- Web dashboard built using HTML and Python Flask to visualize the results 

We used a dataset provided by [Figure Eight](https://www.figure-eight.com/) with real messages that were sent during disaster events. We created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

The goal of the project is to classify the disaster messages into categories. 


## Project Descriptions<a name = "descriptions"></a>
1. **ETL Pipeline:** `process_data.py` file contain the script to create ETL pipline which:

- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. **ML Pipeline:** `train_classifier.py` file contain the script to create ML pipline which:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. **Flask Web App:** the web app enables the user to enter a disaster message, and then view the categories of the message.

 
## Files Descriptions <a name="files"></a>

The files structure is arranged as below:

	- README.md: read me file
	- workspace
		- \app
			- run.py: flask file to run the app
		- \templates
			- master.html: main page of the web application 
			- go.html: result web page
		- \data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
			- data.db: disaster response database
			- process_data.py: ETL process
		- \models
			- train_classifier.py: classification code

## Instructions <a name="instructions"></a>

Follow the steps below to run the project in your local machine:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/data.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/data.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to localhost: http://0.0.0.0:3001/

## Screenshots

![alt text](https://video.udacity-data.com/topher/2018/September/5b967bef_disaster-response-project1/disaster-response-project1.png)
![alt text](https://video.udacity-data.com/topher/2018/September/5b967cda_disaster-response-project2/disaster-response-project2.png)

