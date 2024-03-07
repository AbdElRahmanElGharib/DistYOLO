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
        box_prediction, class_prediction = inputs['boxes'], inputs['classes']

        confidence_prediction = tf.math.reduce_max(class_prediction, axis=-1)

        idx, valid_det = tf.image.non_max_suppression_padded(
            box_prediction,
            confidence_prediction,
            max_output_size=self.max_detections,
            iou_threshold=self.iou_threshold,
            score_threshold=self.confidence_threshold,
            pad_to_max_output_size=True,
            sorted_input=False,
        )
        # TODO: use tf.gather_nd instead of tf.experimental.numpy.take_along_axis
        box_prediction = tf.experimental.numpy.take_along_axis(
            box_prediction, tf.expand_dims(idx, axis=-1), axis=1
        )
        box_prediction = tf.reshape(
            box_prediction, (-1, self.max_detections, 4)
        )
        # TODO: use tf.gather_nd instead of tf.experimental.numpy.take_along_axis
        confidence_prediction = tf.experimental.numpy.take_along_axis(
            confidence_prediction, idx, axis=1
        )
        # TODO: use tf.gather_nd instead of tf.experimental.numpy.take_along_axis
        class_prediction = tf.experimental.numpy.take_along_axis(
            class_prediction, tf.expand_dims(idx, axis=-1), axis=1
        )

        bounding_boxes = {
            "boxes": box_prediction,
            "confidence": confidence_prediction,
            "classes": tf.argmax(class_prediction, axis=-1),
            "num_detections": valid_det,
        }

        return bounding_boxes
