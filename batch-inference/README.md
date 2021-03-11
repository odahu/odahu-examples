

### About

Example shows mechanics of Batch inference

Instead of using some ML Framework it focuses on ODAHU functionality and replaces
"real" model by a simple `model` that contains multiplier.txt that defines a `multiplier`.
Every number in input tensor will be multiplied by `multiplier` and returned as output tensor.

### Example structure:

- `input` defines inference request in [kubeflow format](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md#inference-request-json-object)
- `output` defines expected inference response in [kubeflow format](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md#inference-response-json-object)
- `manifests` defines manifests for ODAHU API
    - `inferenceservice.yaml` is a definition of batch inference service
    - `inferencejob.yaml.yaml` is a definition of batch inference job
- `predictor` defines a predictor image that will be launched by ODAHU system
              during batch inference process. The goal of the predictor is handle input
              and provide output in kubeflow format. 
              Predictor follows ODAHU Batch inference predictor image protocol.
- `model` defines model files
    - `odahuflow.project.yaml` metadata file that is required by ODAHU 
    - `multiplier.txt` file that contains multiplier. This file is used by predictor.py script
- `dag.py` example of using airflow plugin to schedule inference job


### How to

1. Upload `input` files into object storage that is available via `Connection` to ODAHU Cluster
2. Upload `model` files into object storage that is available via `Connection` to ODAHU Cluster
3. Register `inferenceservice.yaml` by making 

```http request
POST /api/v1/batch/service -d @manifests/inferenceservice.yaml
```
4. Launch `inferencejob.yaml` by making 

```http request
POST /api/v1/batch/job -d @manifests/inferencejob.yaml
```

5. Check `InferenceJob.spec.outputDestination.path`. It should contain the same output as `output` folder