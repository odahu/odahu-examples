import argparse
import os
import warnings

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def is_suspicious():
    """For now we just compare against the hard-coded feeder value"""
    # TODO calculate off of something like distance from mean(per drug) by stddev
    return merged['amount'] > 42


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    num_estimators = 2
    random_state = 0
    pickle_size = 8192
    test_pct = .75

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha')
    parser.add_argument('--l1-ratio')

    try:
        args = parser.parse_args()
    except:
        args = argparse.Namespace(alpha=0.1, l1_ratio=0.5)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)

    hl7_path = os.path.join(os.path.dirname(os.path.relpath('__file__')), "test.snappy.parquet/")
    claims_path = os.path.join(os.path.dirname(os.path.relpath('__file__')), "claims.snappy.parquet/")
    #    wine_path = os.path.join(os.path.dirname(os.path.relpath('__file__')),"testfile.snappy.parquet")
    #    df = pd.read_csv(wine_path)

    hl7_df = pd.read_parquet(path=hl7_path, engine='pyarrow')
    claims_df = pd.read_parquet(path=claims_path, engine='pyarrow')

    # DATA PROCESSING

    merged = pd.merge(hl7_df, claims_df, on='patientId', how='inner') \
        .drop(['id'], axis=1)
    merged['amount'] = merged['price'].astype(int)
    # the is_suspicious column is what we will be predicting
    merged['is_suspicious'] = is_suspicious()
    df = merged.drop(['price'], axis=1)
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= test_pct
    factorizedDrugs = pd.factorize(df['drug'])[0]
    factorizedSuspicions = pd.factorize(df['is_suspicious'])[0]
    df.insert(loc=len(df.columns), column='drugId', value=factorizedDrugs)
    df.insert(loc=len(df.columns), column='suspicion_number', value=factorizedSuspicions)
    df['patientId'] = df['patientId'].astype(int)

    train, test = df[df['is_train']], df[df['is_train'] == False]
    df = df.drop(['drug', 'is_suspicious', 'is_train'], axis=1)
    labels = df.copy().pop('suspicion_number')
    train, test = train_test_split(df)


    features = df.columns[0:3]

    train_x = train.drop(["suspicion_number"], axis=1)
    test_x = test.drop(["suspicion_number"], axis=1)
    train_y = train[["suspicion_number"]]
    test_y = test[["suspicion_number"]]

    alpha = float(args.alpha or 1.0)
    l1_ratio = float(args.l1_ratio or 2.0)

    with mlflow.start_run():
        classifier = RandomForestClassifier(n_jobs=num_estimators,
                                        random_state=random_state)
        classifier.fit(train_x, pd.factorize(train['suspicion_number'])[0])

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
