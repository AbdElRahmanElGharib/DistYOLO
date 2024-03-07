import tensorflow as tf
import keras
# import tensorflow.keras as keras
from prediction_decoder import PredictionDecoder


class MeanAveragePrecision(keras.metrics.Metric):
    def __init__(self, *args, **kwargs):
        super(MeanAveragePrecision, self).__init__(*args, **kwargs)

    def result(self):
        pass

    def update_state(self, *args, **kwargs):
        pass
