# Disaster Response Pipeline Project 
Udacity project

## Introduction
The data provided by FigureEight/Appen(https://appen.com/) uses data containing real messages that were sent during disaster events.
The aim of this project is to train an ML model to categorize these events.
This project includes the machine learning pipeline and a web application that can be used to classify new input messages.

### Instructions
Pipeline has been tested with python 3.8

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Structure
- app - web app files
    - run.py 
- data - data and script to process it for classification
    - disaster_messages.csv
    - disaster_categories.csv
    - process_data.py
- models - contains pkl files and script to train the model
    - train_classifier.py

## Notes
The data is very imbalanced. See figure below:
![alt text](https://github.com/sukiat11/disaster-response-project/blob/main/images/class_distribution.png)


A machine learning model that can handle this situation better would be a boosting algorithm. For example, the adaboost algorithm performed better
than the linear svc algorithm. It handles the issue by maintaining a set of weights on the training data set in the learning process.

The category "child_alone" was not included in training since all its values were the same. Also all categories' values were mapped to binary,
as this is the expected format. Any values of 2 were set to 1.

## Results
![alt text](https://github.com/sukiat11/disaster-response-project/blob/main/images/report_adaboost.png)








