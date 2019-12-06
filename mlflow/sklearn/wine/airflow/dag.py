from datetime import datetime

from airflow import DAG
from airflow.contrib.operators.gcs_to_gcs import GoogleCloudStorageToGoogleCloudStorageOperator
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator
from odahuflow.sdk.models import ModelTraining, ModelTrainingSpec, ModelIdentity, ResourceRequirements, ResourceList, \
    ModelPackaging, ModelPackagingSpec, Target, ModelDeployment, ModelDeploymentSpec, Connection, ConnectionSpec, \
    DataBindingDir

from odahuflow.airflow_plugin.connection import GcpConnectionToOdahuConnectionOperator
from odahuflow.airflow_plugin.deployment import DeploymentOperator, DeploymentSensor
from odahuflow.airflow_plugin.model import ModelPredictRequestOperator, ModelInfoRequestOperator
from odahuflow.airflow_plugin.packaging import PackagingOperator, PackagingSensor
from odahuflow.airflow_plugin.training import TrainingOperator, TrainingSensor

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 9, 3),
    'email_on_failure': False,
    'email_on_retry': False,
    'end_date': datetime(2099, 12, 31)
}

api_connection_id = "odahuflow_api"
model_connection_id = "odahuflow_model"

gcp_project = Variable.get("GCP_PROJECT")
wine_bucket = Variable.get("WINE_BUCKET")

wine_conn_id = "wine"
wine = Connection(
    id=wine_conn_id,
    spec=ConnectionSpec(
        type="gcs",
        uri=f'gs://{wine_bucket}/data/wine-quality.csv',
        region=gcp_project,
    )
)

training_id = "airlfow-wine"
training = ModelTraining(
    id=training_id,
    spec=ModelTrainingSpec(
        model=ModelIdentity(
            name="wine",
            version="1.0"
        ),
        toolchain="mlflow",
        entrypoint="main",
        work_dir="mlflow/sklearn/wine",
        hyper_parameters={
            "alpha": "1.0"
        },
        data=[
            DataBindingDir(
                conn_name='wine',
                local_path='mlflow/sklearn/wine/wine-quality.csv'
            ),
        ],
        resources=ResourceRequirements(
            requests=ResourceList(
                cpu="2024m",
                memory="2024Mi"
            ),
            limits=ResourceList(
                cpu="2024m",
                memory="2024Mi"
            )
        ),
        vcs_name="odahuflow-examples"
    ),
)

packaging_id = "airlfow-wine"
packaging = ModelPackaging(
    id=packaging_id,
    spec=ModelPackagingSpec(
        targets=[Target(name="docker-push", connection_name="docker-ci")],
        integration_name="docker-rest"
    ),
)

deployment_id = "airlfow-wine"
deployment = ModelDeployment(
    id=deployment_id,
    spec=ModelDeploymentSpec(
        min_replicas=1,
    ),
)

model_example_request = {
    "columns": ["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH",
                "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],
    "data": [[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66],
             [12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]
}

dag = DAG(
    'wine_model',
    default_args=default_args,
    schedule_interval=None
)

with dag:
    data_extraction = GoogleCloudStorageToGoogleCloudStorageOperator(
        task_id='data_extraction',
        google_cloud_storage_conn_id='wine_input',
        source_bucket=wine_bucket,
        destination_bucket=wine_bucket,
        source_object='input/*.csv',
        destination_object='data/',
        project_id=gcp_project,
        default_args=default_args
    )
    data_transformation = BashOperator(
        task_id='data_transformation',
        bash_command='echo "imagine that we transform a data"',
        default_args=default_args
    )
    odahuflow_conn = GcpConnectionToOdahuConnectionOperator(
        task_id='odahuflow_connection_creation',
        google_cloud_storage_conn_id='wine_input',
        api_connection_id=api_connection_id,
        conn_template=wine,
        default_args=default_args
    )

    train = TrainingOperator(
        task_id="training",
        api_connection_id=api_connection_id,
        training=training,
        default_args=default_args
    )

    wait_for_train = TrainingSensor(
        task_id='wait_for_training',
        training_id=training_id,
        api_connection_id=api_connection_id,
        default_args=default_args
    )

    pack = PackagingOperator(
        task_id="packaging",
        api_connection_id=api_connection_id,
        packaging=packaging,
        trained_task_id="wait_for_training",
        default_args=default_args
    )

    wait_for_pack = PackagingSensor(
        task_id='wait_for_packaging',
        packaging_id=packaging_id,
        api_connection_id=api_connection_id,
        default_args=default_args
    )

    dep = DeploymentOperator(
        task_id="deployment",
        api_connection_id=api_connection_id,
        deployment=deployment,
        packaging_task_id="wait_for_packaging",
        default_args=default_args
    )

    wait_for_dep = DeploymentSensor(
        task_id='wait_for_deployment',
        deployment_id=deployment_id,
        api_connection_id=api_connection_id,
        default_args=default_args
    )

    model_predict_request = ModelPredictRequestOperator(
        task_id="model_predict_request",
        model_deployment_name=deployment_id,
        api_connection_id=api_connection_id,
        model_connection_id=model_connection_id,
        request_body=model_example_request,
        default_args=default_args
    )

    model_info_request = ModelInfoRequestOperator(
        task_id='model_info_request',
        model_deployment_name=deployment_id,
        api_connection_id=api_connection_id,
        model_connection_id=model_connection_id,
        default_args=default_args
    )

    data_extraction >> data_transformation >> odahuflow_conn >> train
    train >> wait_for_train >> pack >> wait_for_pack >> dep >> wait_for_dep
    wait_for_dep >> model_info_request
    wait_for_dep >> model_predict_request
