# The data set used in this example is generated from a pipeline Feeder
# defined here: https://github.com/jasonnerothin/hl7fwa and ingested into
# two GCS shares defined in legion/

import argparse
import os
import warnings

import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    #    wine_path = os.path.join(os.path.dirname(os.path.relpath('__file__')),"testfile.snappy.parquet")
    #    df = pd.read_csv(wine_path)

    iris = load_iris()
    print(iris)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add a new column with the species names, this is what we are going to try to predict
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print('With species...')
    print(df.head())

    # Create a new column that for each row, generates a random number between 0 and 1, and
    # if that value is less than or equal to .75, then sets the value of that cell as True
    # and false otherwise. This is a quick and dirty way of randomly assigning some rows to
    # be used as the training data and some as the test data.
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    print('With is_train...')
    print(df.head())
    # Create two new dataframes, one with the training rows, one with the test rows
    train, test = df[df['is_train'] == True], df[df['is_train'] == False]
    # Show the number of observations for the test and training dataframes
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))
    # Create a list of the feature column's names
    features = df.columns[:4]  # LOOK HERE
    print('Feature names:')
    print(features)
    # train['species'] contains the actual species names. Before we can use it,
    # we need to convert each species name into a digit. So, in this case there
    # are three species, which have been coded as 0, 1, or 2.
    y = pd.factorize(train['species'])[0]
    print('Make species numeric')
    print(y)

    #    alpha = float(args.alpha or 1.0)
    #    l1_ratio = float(args.l1_ratio or 2.0)

    with mlflow.start_run():
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        print('Fitting...')
        clf.fit(train[features], y)
        print('Fit.')

        print('Predicting...')
        clf.predict(test[features])
        print('Predicted.')
        print(clf.predict_proba(test[features])[0:10])

        ##        (rmse, mae, r2) = (1.0,2.0,3.0)  # eval_metrics(test_y, predicted_qualities)

        #        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        #        print(f"  RMSE: {rmse}")
        #        print(f"  MAE: {mae}")
        #        print(f"  R2: {r2}")

        ##        mlflow.log_param("alpha", alpha)
        ##        mlflow.log_param("l1_ratio", l1_ratio)
        ##        mlflow.log_metric("rmse", rmse)
        ##        mlflow.log_metric("r2", r2)
        ##        mlflow.log_metric("mae", mae)
        log_id = 'randomforest5'

        mlflow.sklearn.log_model(clf, log_id)

        # Persist samples (input and output)
        train.head().to_pickle('head_input.pkl')
        mlflow.log_artifact('head_input.pkl', log_id)
        test.head().to_pickle('head_output.pkl')
        mlflow.log_artifact('head_output.pkl', log_id)
