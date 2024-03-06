import tensorflow as tf
import keras
# import tensorflow.keras as keras
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_label_encoder import YOLOV8LabelEncoder as LabelEncoder


def is_anchor_center_within_box(anchors, gt_bboxes):
    return tf.math.reduce_all(
        tf.math.logical_and(
            gt_bboxes[:, :, None, :2] < anchors,
            gt_bboxes[:, :, None, 2:] > anchors,
        ),
        axis=-1,
    )


class LabelEncoder(keras.layers.Layer):
    def __init__(
            self,
            num_classes,
            max_anchor_matches=10,
            alpha=0.5,
            beta=6.0,
            epsilon=1e-9,
            **kwargs
    ):
        super(LabelEncoder, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.max_anchor_matches = max_anchor_matches
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def call(self, inputs, *args, **kwargs):
        scores = inputs['scores']
        decode_bboxes = inputs['decode_bboxes']
        anchors = inputs['anchors']
        gt_labels = inputs['gt_labels']
        gt_bboxes = inputs['gt_bboxes']
        gt_mask = inputs['gt_mask']
