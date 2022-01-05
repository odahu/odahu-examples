#  Copyright 2021 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

'''
1. make sure the dataset is present in bucket
  -

2.
{"data_bucket": "odahu-ci-data-store"}
'''
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from odahuflow.sdk.models import ModelTraining, ModelTrainingSpec, ModelIdentity, ResourceRequirements, ResourceList, \
    ModelPackaging, ModelPackagingSpec, ModelDeployment, ModelDeploymentSpec, DataBindingDir, AlgorithmSource, VCS
from odahuflow.airflow_plugin.deployment import DeploymentOperator, DeploymentSensor
from odahuflow.airflow_plugin.packaging import PackagingOperator, PackagingSensor
from odahuflow.airflow_plugin.training import TrainingOperator, TrainingSensor


API_CONNECTION_ID = 'odahuflow_api'


dataset_local_filename = 'dogs_cats.zip'
dataset_bucket_path = f'input/{dataset_local_filename}'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 9, 3),
    'email_on_failure': False,
    'email_on_retry': False,
}

with DAG(
    dag_id='dogs_cats_scenario',
    default_args=default_args,
    schedule_interval=None,
    tags=['example'],
) as dag:

    def callable_virtualenv(*args, dag_run, **kwargs):
        """
        Example function that will be performed in a virtual environment.

        Importing at the module level ensures that it will not attempt to import the
        library before it is installed.
        """
        from urllib.request import urlretrieve
        import os
        import shutil
        from airflow.contrib.hooks.gcs_hook import GoogleCloudStorageHook

        run_config = dag_run.conf

        airflow_data_connection = run_config.get('airflow_data_connection', 'wine_input')
        bucket_name = run_config.get('data_bucket', 'odahu-dev03-data-store')

        tmp_files = set()

        client = GoogleCloudStorageHook(google_cloud_storage_conn_id=airflow_data_connection)
        if client.exists(bucket_name, dataset_bucket_path):
            print(f'File {dataset_bucket_path} already exists in bucket {bucket_name}')
            return

        dataset_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/' \
                      'kagglecatsanddogs_3367a.zip'
        urlretrieve(dataset_url, filename=dataset_local_filename)
        tmp_files.add(dataset_local_filename)

        client.upload(bucket=bucket_name, object=dataset_bucket_path, filename=dataset_local_filename)

        for file in tmp_files:
            print(f'Clean-up: {file}')
            if os.path.isdir(file):
                shutil.rmtree(file)
            os.remove(file)


    ensure_dataset_in_bucket = PythonOperator(
        task_id='ensure_dataset_in_bucket',
        provide_context=True,
        python_callable=callable_virtualenv,
    )

    training_id = 'dogs-cats-classifier'
    training = ModelTraining(
        id=training_id,
        spec=ModelTrainingSpec(
            model=ModelIdentity(
                name='dogs-cats-classifier',
                version='1'
            ),
            toolchain='mlflow-project',
            entrypoint='main',
            work_dir='mlflow/keras/dogs_cats',
            hyper_parameters={
                'epochs': '50'
            },
            data=[
                DataBindingDir(
                    connection='models-output',
                    remote_path=dataset_bucket_path,
                    local_path=f'mlflow/keras/dogs_cats/{dataset_local_filename}'
                ),
            ],
            resources=ResourceRequirements(
                requests=ResourceList(
                    cpu='4',
                    memory='16Gi',
                    gpu='1'
                ),
                limits=ResourceList(
                    cpu='4',
                    memory='16Gi',
                    gpu='1'
                ),
            ),
            algorithm_source=AlgorithmSource(vcs=VCS(
                connection='odahu-flow-examples',
                # TODO: switch to develop
                reference='develop'
            ))
        ),
    )

    train = TrainingOperator(
        task_id='submit_training',
        api_connection_id=API_CONNECTION_ID,
        training=training,
    )

    wait_for_train = TrainingSensor(
        task_id='training',
        training_id=training_id,
        api_connection_id=API_CONNECTION_ID,
        timeout=60 * 60 * 4,  # 4 hours
    )

    packaging_id = 'dogs-cats-classifier'
    packaging = ModelPackaging(
        id=packaging_id,
        spec=ModelPackagingSpec(
            integration_name='docker-triton',
            resources=ResourceRequirements(
                requests=ResourceList(
                    cpu='1',
                    memory='4Gi',
                ),
                limits=ResourceList(
                    cpu='2',
                    memory='8Gi',
                ),
            ),
        ),
    )

    pack = PackagingOperator(
        task_id='submit_packaging',
        api_connection_id=API_CONNECTION_ID,
        packaging=packaging,
        trained_task_id='training',
    )

    wait_for_pack = PackagingSensor(
        task_id='packaging',
        packaging_id=packaging_id,
        api_connection_id=API_CONNECTION_ID,
    )

    deployment_id = 'dogs-cats-classifier'
    deployment = ModelDeployment(
        id=deployment_id,
        spec=ModelDeploymentSpec(
            min_replicas=1,
            predictor='triton',
            resources=ResourceRequirements(
                requests=ResourceList(
                    gpu='1'
                ),
            ),
        ),
    )

    dep = DeploymentOperator(
        task_id='submit_deployment',
        api_connection_id=API_CONNECTION_ID,
        deployment=deployment,
        packaging_task_id='packaging',
    )

    wait_for_dep = DeploymentSensor(
        task_id='deployment',
        deployment_id=deployment_id,
        api_connection_id=API_CONNECTION_ID,
    )

    ensure_dataset_in_bucket >> train >> wait_for_train >> pack >> wait_for_pack \
    >> dep >> wait_for_dep
