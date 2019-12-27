from datetime import datetime

from airflow import DAG
from odahuflow.sdk.models import ModelTraining, ModelTrainingSpec, ModelIdentity, ResourceRequirements, ResourceList, \
    ModelPackaging, ModelPackagingSpec, Target, ModelDeployment, ModelDeploymentSpec

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

training_id = "airflow-tensorflow"
training = ModelTraining(
    id=training_id,
    spec=ModelTrainingSpec(
        model=ModelIdentity(
            name="tensorflow",
            version="1.0"
        ),
        toolchain="mlflow",
        entrypoint="main",
        work_dir="mlflow/tensorflow/example",
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
        vcs_name="odahu-flow-examples"
    ),
)

packaging_id = "airflow-tensorflow"
packaging = ModelPackaging(
    id=packaging_id,
    spec=ModelPackagingSpec(
        targets=[Target(name="docker-push", connection_name="docker-ci")],
        integration_name="docker-rest"
    ),
)

deployment_id = "airflow-tensorflow"
deployment = ModelDeployment(
    id=deployment_id,
    spec=ModelDeploymentSpec(
        min_replicas=1,
    ),
)

model_example_request = {
    "columns": ["features"] * 13,
    "data": [
        [18.08460, 0.0, 18.10, 0.0, 0.6790, 6.434, 100.0, 1.8347, 24.0, 666.0, 20.2, 27.25, 29.05],
        [2.92400, 0.0, 19.58, 0.0, 0.6050, 6.101, 93.0, 2.2834, 5.0, 403.0, 14.7, 240.16, 9.81]
    ]
}

dag = DAG(
    'airflow-tensorflow',
    default_args=default_args,
    schedule_interval=None
)

with dag:
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

    train >> wait_for_train >> pack >> wait_for_pack >> dep >> wait_for_dep
    wait_for_dep >> model_info_request
    wait_for_dep >> model_predict_request
