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
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import dask.dataframe as dd


log_id = 'hl7fwa-randomforest'


def parse_input_arguments():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-estimators')
    parser.add_argument('--random-state')
    parser.add_argument('--pickle-size')
    parser.add_argument('--test-pct')
    try:
        args = parser.parse_args()
    except:
        args = argparse.Namespace(num_estimators=2, random_state=40, pickle_size=8192, test_pct=.75)


def is_suspicious():
    """For now we just compare against the hard-coded feeder value"""
    # TODO calculate off of something like distance from mean(per drug) by stddev
    return merged['amount'] > 42


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    parse_input_arguments()

    num_estimators = int(args.num_estimators or 2)
    random_state = int(args.random_state or 0)
    pickle_size = int(args.pickle_size or 8192)
    test_pct = float(args.test_pct or .75)

    # DATA LOAD
    # test_data_dir = '/Users/jason/scratch/hl7fwa-testdata/'
    # hl7_path = os.path.join(test_data_dir, 'kafka_example_waste.parquet')
    # claims_path = os.path.join(test_data_dir, 'pg_example_waste.parquet')

    # TODO: read training data from GCS instead
    hl7_path = os.path.join(os.path.dirname(os.path.relpath('__file__')),"test.snappy.parquet")
    claims_path = os.path.join(os.path.dirname(os.path.relpath('__file__')),"testfile.snappy.parquet")
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
    df.insert(loc=len(df.columns), column='drug_number', value=factorizedDrugs)
    df.insert(loc=len(df.columns), column='suspicion_number', value=factorizedSuspicions)
    train, test = df[df['is_train']], df[df['is_train'] == False]
    df = df.drop(['drug', 'is_suspicious', 'is_train'], axis=1)
    labels = df.copy().pop('suspicion_number')
    x, y, train_labels, test_labels = train_test_split(df, labels, stratify=labels, test_size=test_pct, random_state=random_state)

    # Show the size of test and training dfs
    print('Training data count:', len(train))
    print('Test data count:', len(test))
    features = df.columns[0:4]
    print('Feature names: {}'.format(features))

    with mlflow.start_run():
        classifier = RandomForestClassifier(n_jobs=num_estimators,
                                            random_state=random_state)

        # TODO these could go into MLProject as configurable parameters
        # class_weight=None, criterion='gini', max_depth=None,
        # max_features=None, max_leaf_nodes=None,
        # min_impurity_decrease=0.0, min_impurity_split=None,
        # min_samples_leaf=1, min_samples_split=2,
        # min_weight_fraction_leaf=0.0, presort=False, random_state=50,
        # splitter='best'

        classifier.fit(train[features], pd.factorize(train['suspicion_number'])[0])

        train_preds = classifier.predict(train[features])
        test_preds = classifier.predict(test[features])
        train_probs = classifier.predict_proba(train[features])
        test_probs = classifier.predict_proba(test[features])

        print(len(test_probs))

        crosstab = pd.crosstab(test['suspicion_number'],
                               test_preds,
                               rownames=['Actual Suspicions'],
                               colnames=['Predicted Suspicions'])

        print(crosstab)

        mlflow.log_param("num_estimators", num_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("pickle_size", pickle_size)
        mlflow.log_param("test_pct", test_pct)

        mlflow.sklearn.log_model(classifier, log_id)

        # Model metrics
        mlflow.log_metric("baseline_recall", recall_score(test_labels, [1 for _ in range(len(test_labels))]))
        mlflow.log_metric("baseline_precision", precision_score(test_labels, [1 for _ in range(len(test_labels))]))
        mlflow.log_metric("baseline_roc", 0.5)

        # mlflow.log_metric("results_recall", recall_score(test_labels, test_preds))
        # mlflow.log_metric("results_precision", precision_score(test_labels, test_preds))
        # mlflow.log_metric("results_roc", roc_auc_score(test_labels, test_probs))
        # mlflow.log_metric("train_results_recall", recall_score(labels, train_preds))
        # mlflow.log_metric("train_results_precision",  precision_score(labels, train_preds))
        # mlflow.log_metric("train_results_roc", roc_auc_score(labels, train_probs))
        # Calculate false positive rates and true positive rates
        # base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
        # model_fpr, model_tpr, _ = roc_curve(test_labels, test_probs)
        #
        # mlflow.log_metric("base_fpr", base_fpr)
        # mlflow.log_metric("base_tpr", base_tpr)
        # mlflow.log_metric("model_fpr", model_fpr)
        # mlflow.log_metric("model_tpr", model_tpr)

        # Persist sample data (input and output)
        train.head(pickle_size).to_pickle('head_input.pkl')
        mlflow.log_artifact('head_input.pkl', log_id)
        test.head(pickle_size).to_pickle('head_output.pkl')
        mlflow.log_artifact('head_output.pkl', log_id)
