import argparse
import os
import warnings

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    #  pd.set_option('display.max_columns', 500)

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha')
    parser.add_argument('--l1-ratio')

    try:
        args = parser.parse_args()
    except:
        args = argparse.Namespace(alpha=0.1, l1_ratio=0.5)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)

    hl7_path = os.path.join(os.path.dirname(os.path.relpath('__file__')), "result")
    data_dir = Path(hl7_path)
    full_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in data_dir.glob('*.parquet')
    )

    full_df["PotentialFraud"] = full_df['PotentialFraud'].apply(lambda x: 0 if x == 'No' else 1)

    X = full_df.drop(axis=1, columns=['Provider', 'PotentialFraud'])
    y = full_df['PotentialFraud']

    print(full_df.head(4))

    train, test = train_test_split(full_df)

    train_x = train.drop(['Provider', 'PotentialFraud'], axis=1)
    test_x = test.drop(['Provider', 'PotentialFraud'], axis=1)
    train_y = train[["PotentialFraud"]]
    test_y = test[["PotentialFraud"]]

    train_x = train_x.fillna(0)
    test_x = test_x.fillna(0)
    train_y = train_y.fillna(0)
    test_y = test_y.fillna(0)
    alpha = float(args.alpha or 1.0)
    l1_ratio = float(args.l1_ratio or 2.0)

    with mlflow.start_run():
        classifier = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=123,
                                            max_depth=4)  # We will set max_depth =4
        classifier.fit(train_x, train_y)

        predicted_qualities = classifier.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(classifier, "model")

        # Persist samples (input and output)
        train_x.head().to_pickle('head_input.pkl')
        mlflow.log_artifact('head_input.pkl', 'model')
        train_y.head().to_pickle('head_output.pkl')
        mlflow.log_artifact('head_output.pkl', 'model')