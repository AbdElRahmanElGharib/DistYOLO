import tensorflow as tf
import keras
# import tensorflow.keras as keras


class NonMaxSuppression(keras.layers.Layer):
    def __init__(
        self,
        iou_threshold=0.5,
        confidence_threshold=0.5,
        max_detections=100,
        **kwargs,
    ):
        super(NonMaxSuppression, self).__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.built = True

    def call(self, inputs, *args, **kwargs):
        pass
