How To Train and Deploy Image Classifier with MLflow and Keras
---------------------------------------------------------------

In this example we demonstrate how to train and deploy image classification models with MLflow.
We train a VGG16 deep learning model to classify flower species from photos using a `dataset
<http://download.tensorflow.org/example_images/flower_photos.tgz>`_ available from `tensorflow.org
<http://www.tensorflow.org>`_. Note that although we use Keras to train the model in this case,
a similar approach can be applied to other deep learning frameworks such as ``PyTorch``.

The MLflow model produced by running this example can be deployed to any MLflow supported endpoints.
All the necessary image preprocessing is packaged with the model. The model can therefore be applied
to image data directly. All that is required in order to pass new data to the model is to encode the
image binary data as base64 encoded string in pandas DataFrame (standard interface for MLflow python
function models). The included Python scripts demonstrate how the model can be deployed to a REST
API endpoint for realtime evaluation or to Spark for batch scoring..

In order to include custom image pre-processing logic with the model, we define the model as a
custom python function model wrapping around the underlying Keras model. The wrapper provides
necessary preprocessing to convert input data into multidimensional arrays expected by the
Keras model. The preprocessing logic is stored with the model as a code dependency. Here is an
example of the output model directory layout:

.. code-block:: bash

   tree model

::

   model
   ├── MLmodel
   ├── code
   │   └── image_pyfunc.py
   ├── data
   │   └── image_model
   │       ├── conf.yaml
   │       └── keras_model
   │           ├── MLmodel
   │           ├── conda.yaml
   │           └── model.h5
   └── mlflow_env.yml



The example contains the following files:

 * MLproject
   Contains definition of this project. Contains only one entry point to train the model.

 * conda.yaml
   Defines project dependencies. NOTE: You might want to change tensorflow package to tensorflow-gpu
   if you have gpu(s) available.

 * train.py
   Main entry point of the projects. Handles command line arguments and possibly downloads the
   dataset.

 * image_pyfunc.py
   The implementation of the model train and also of the outputed custom python flavor model. Note
   that the same preprocessing code that is used during model training is packaged with the output
   model and is used during scoring.

 * score_images_rest.py
   Score an image or a directory of images using a model deployed to a REST endpoint.

 * score_images_spark.py
   Score an image or a directory of images using model deployed to Spark.



Running this Example
^^^^^^^^^^^^^^^^^^^^

To train the model, run the example as a standard MLflow project:


.. code-block:: bash

    mlflow run examples/flower_classifier

This will download the training dataset from ``tensorflow.org``, train a classifier using Keras and
log results with Mlflow.

To test your model, run the included scoring scripts. For example, say your model was trained with
run_id ``101``.

- To test REST api scoring do the following two steps:

  1. Deploy the model as a local REST endpoint by running ``mlflow models serve``:

  .. code-block:: bash

      # deploy the model to local REST api endpoint
      mlflow models serve --model-uri runs:/101/model --port 54321


  2. Apply the model to new data using the provided score_images_rest.py script:

  .. code-block:: bash

      # score the deployed model
      python score_images_rest.py --model-uri runs:/101/model --port 54321 http://127.0.0.1 --data-path /path/to/images/for/scoring


- To test batch scoring in Spark, run score_images_spark.py to score the model in Spark like this:

.. code-block:: bash

   python score_images_spark.py --model-uri runs:/101/model --data-path /path/to/images/for/scoring


How to predict flower class using image
------------------------------------------

In order to get prediction from trained and deployed in ODAHU cluster flower model you should

  1. Encode flower image in base64 and save it into json payload for api
     (You can also `download original dataset <http://download.tensorflow.org/example_images/flower_photos.tgz>`_)

  .. code-block:: console

     $ jq -n --arg encoded_image "$(base64 odahuflow/tulip.jpg)" '{"columns": ["image"], "data": [[$encoded_image]]}' > odahuflow/request.json

  2. Make request to API (Do not forget login previously using odahuflow login --url <cluster-url>)

  .. code-block:: console

     $ odahuflowctl model invoke --md=flower-classifier --json-file=odahuflow/request.json | jq

  3. You can see result like this

  .. code-block:: json

    {
      "prediction": [
        [
          "tulips",
          "4",
          "0.05327853",
          "0.23025948",
          "0.23088738",
          "0.22562687",
          "0.2599477"
        ]
      ],
      "columns": [
        "predicted_label",
        "predicted_label_id",
        "p(sunflowers)",
        "p(dandelion)",
        "p(roses)",
        "p(daisy)",
        "p(tulips)"
      ]
    }



