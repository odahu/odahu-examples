from typing import List

import mlflow.pyfunc
import pandas as pd
from keras import models as keras_models
from keras.preprocessing.text import Tokenizer
import keras.backend
import tensorflow as tf
from mlflow.pyfunc import PythonModelContext
from sklearn.preprocessing import MultiLabelBinarizer

from src import parser

def fit_tokenizer(streamer: parser.ReutersStreamer, max_words) -> Tokenizer:
    # Tokenize all reuters dataset

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(
        (text for text, _ in streamer.stream_reuters_documents_with_topics())
    )
    return tokenizer


def fit_topics_encoder(streamer: parser.ReutersStreamer) -> MultiLabelBinarizer:

    topics_encoder = MultiLabelBinarizer()
    topics_encoder.fit(
        ([t] for t in streamer.topics)
    )
    return topics_encoder


def float_to_percentage(value):
    n_value = round(value, 4) * 100
    return "{0:.2f}%".format(n_value)


def save_samples(topics: List[str]) -> List[str]:
    """
    Save samples of data input and output. Return paths to pickled files
    :param topics:
    :return:
    """

    input_fp = 'head_input.pkl'
    output_fp = 'head_output.pkl'

    pd.DataFrame(data={'text': [
        '''
        The Bank of England said it had provided the
        money market with a further 437 mln stg assistance in the
        afternoon session. This brings the Bank's total help so far
        today to 461 mln stg and compares with its revised shortage
        forecast of 450 mln stg.
            The central bank made purchases of bank bills outright
        comprising 120 mln stg in band one at 10-7/8 pct and 315 mln
        stg in band two at 10-13/16 pct.
            In addition, it also bought two mln stg of treasury bills
        in band two at 10-13/16 pct.
        '''
    ]}).to_pickle(input_fp)

    pd.DataFrame(columns=topics, data=[[
        "{0:.2f}%".format(42.12) for i in topics
    ]]).to_pickle(output_fp)

    return [input_fp, output_fp]


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """
    We define wrapper for keras model to change interface of keras model by adding "encoding input data"
    and "decoding output data" features.
    Without this
    """

    def __init__(self, tokenizer: Tokenizer, topics_encoder: MultiLabelBinarizer):
        self.graph = None
        self.sess = None
        self.tokenizer = tokenizer  # tokenizer that memorized word index from text corpus used in training step
        self.topics_encoder = topics_encoder  # Multi label topics encoder
        self.keras_model = None

    def load_context(self, context: PythonModelContext):
        if 'keras_model' not in context.artifacts:
            raise RuntimeError('"keras_model" artifact is not found. It should be stored keras model as h5 file')
        model_path = context.artifacts['keras_model']

        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with graph.as_default():
            with sess.as_default():  # pylint:disable=not-context-manager
                keras.backend.set_learning_phase(0)
                self.keras_model = keras_models.load_model(model_path)
                self.sess = sess
                self.graph = graph

    def predict(self, context, model_input: pd.DataFrame):
        X = self.tokenizer.texts_to_matrix(model_input['text'])
        with self.graph.as_default():
            with self.sess.as_default():
                Y = pd.DataFrame(self.keras_model.predict(X))
        Y.index = model_input.index
        Y.columns = self.topics_encoder.classes_
        Y = Y.applymap(float_to_percentage)
        return Y