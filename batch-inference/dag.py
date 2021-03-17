#
#      Copyright 2021 EPAM Systems
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
import os
from datetime import datetime
from odahuflow.airflow_plugin.batch import InferenceJobOperator, InferenceJobSensor, InferenceServiceOperator
from odahuflow.airflow_plugin.resources import resource
from pathlib import Path

from airflow import DAG

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 9, 3),
    'email_on_failure': False,
    'email_on_retry': False,
    'end_date': datetime(2099, 12, 31)
}

dag = DAG(
    'batch-example',
    default_args=default_args,
    schedule_interval=None,
)

with dag:

    _, service_payload = resource(Path(os.path.dirname(__file__)) / "manifests/inferenceservice.yaml")
    service = InferenceServiceOperator(
        task_id="inference-service",
        service=service_payload,
        api_connection_id="odahu",
        default_args=default_args
    )
    _, job_payload = resource(Path(os.path.dirname(__file__)) / "manifests/inferencejob.yaml")
    job = InferenceJobOperator(
        task_id="inference-job",
        job=job_payload, api_connection_id="odahu",
        default_args=default_args,
    )

    job_sensor = InferenceJobSensor(
        task_id="wait-inference-job",
        inference_job_task_id="inference-job",
        api_connection_id="odahu",
        default_args=default_args
    )

    service >> job >> job_sensor

if __name__ == '__main__':
    dag.clear(reset_dag_runs=True)
    dag.run(donot_pickle=True)
