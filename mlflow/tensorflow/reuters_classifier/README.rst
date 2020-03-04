Reuters classification for ODAHU
====================================

What's problem the model solve?
-----------------------------------------

The example shows how to solve document classification problem using ODAHU.

Document classification or document categorization is a problem in library science, information science and computer science.
The task is to assign a document to one or more classes or categories

Well-known  `Reuters-21578 dataset <https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html>`_ is used for training the model.

The purpose of this example is not to show best approach to solve the document classification problem but rather
to show features of MLFlow library.

`Keras Deep learning library <https://keras.io/>`_ is used to train model.

Structure of example is shown below

.. code-block:: bash

   tree reuters_classifier

::

    reuters_classifier/
    ├── MLproject
    ├── README.rst
    ├── conda.yaml
    ├── odahuflow
    │   ├── deployment.odahuflow.yaml
    │   ├── git_connection.odahuflow.yaml
    │   ├── packaging.odahuflow.yaml
    │   └── training.odahuflow.yaml
    ├── src
    │   ├── parser.py
    │   └── utils.py
    └── train.py



The example contains the following files:

 * MLproject
   Contains definition of this project. Contains only one entry point to train the model.

 * conda.yaml
   Defines project dependencies.

 * train.py
   Main entry point of the projects.

 * odahuflow
   directory with manifests to interact with ODAHU cluster

 * src/parser.py
   Contain classes to download and parse reuters dataset

 * src/utils.py
   Contain set of help functions and :ref:`ModelWrapper` class, that allow to add extra functionality to model


Because of, in general, deep learning libraries require matrix of number features, we should tokenize text corpus, and encode labels that try to predict.

Word tokenizer and label encoder should also be packed as artifacts with the model because we should encode data
that model receive for predictions

Brief information about train.py script
-----------------------------------------

  - On lines 40-41 we fit tokenizer to get corpus dictionary and encode document labels
  - On lines 44-49 we use fitted tokenizer and encoder to encode features of dataset
  - On line 74 we enable auto logs of metrics for keras
  - On line 84 we store serialized trained model
  - On lines 92-95 we log our model as MLFlow artifact, using ``pyfunc`` flavor.
    You can find more information in `MLFlow docs <https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.log_model>`_
    We log wrapper that contain keras serialized model and extra files such as tokenizer and encoder. For more information see :ref:`ModelWrapper`

How to use the multiple python scripts in a mlflow project?
------------------------------------------------------------

Very often our model require import some other python files to make predictions.

If you just log only model object using MLFlow the ``ImportError`` could be raised in case when you will try
to call this model for prediction during the deserialization phase.
To avoid this problem in ``train.py`` file on line 94 we pass all extra files directory as ``code_path`` parameter.
This is a list of local filesystem paths to Python file dependencies (or directories containing file dependencies).
These files are prepended to the system path before the model is loaded.


ModelWrapper
--------------

Simple logging serialized keras trained model is not enough to make predictions.
Because of we need to have the same word dictionary (tokenizer) for documents which are sent to predict label.
Process of tokenizing could be separated from executables what train model and make predictions.
But in our case we want to encapsulate all this logic into the model.

Therefore model will accept matrix of numbers as input in contrast of trained keras model but pure document text and return human readable label.
To achieve this goal we need to use the same tokenizer and encoder inside the model during the prediction.

Look at ``utils.py`` file, ModelWrapper is defined. This class is accept tokenizer and encoder as regular parameters and set them as attribute values.
As a result these object will be serialized with the ModelWrapper object itself during ``pyfunc`` logs model.

Keras model was saved using keras mechanism of model serialization and it is loaded in ``load_context`` function.

``predict`` method of ``ModelWrapper`` wraps predict api of serialized keras model to add tokenizing input data
and assign human friendly labels to prediction


Read more information about model wrapper in `MLFlow docs <https://www.mlflow.org/docs/latest/models.html#example-saving-an-xgboost-model-in-mlflow-format>`_
