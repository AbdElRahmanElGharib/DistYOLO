import tensorflow as tf
import keras
# import tensorflow.keras as keras
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_label_encoder import YOLOV8LabelEncoder as LaE


def is_anchor_center_within_box(anchors, gt_bboxes):
    return tf.math.reduce_all(
        tf.math.logical_and(
            gt_bboxes[:, :, None, :2] < anchors,
            gt_bboxes[:, :, None, 2:] > anchors,
        ),
        axis=-1,
    )


# class LabelEncoder(keras.layers.Layer):
class LabelEncoder(LaE):
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

        if isinstance(gt_bboxes, tf.RaggedTensor):
            dense_bounding_boxes = bounding_box.to_dense(
                {"boxes": gt_bboxes, "classes": gt_labels},
            )
            gt_bboxes = dense_bounding_boxes["boxes"]
            gt_labels = dense_bounding_boxes["classes"]

        if isinstance(gt_mask, tf.RaggedTensor):
            gt_mask = gt_mask.to_tensor()

        max_num_boxes = tf.shape(gt_bboxes)[1]

        # If there are no GT boxes in the batch, we short-circuit and return
        # empty targets to avoid NaNs.
        return tf.cond(
            tf.constant(max_num_boxes > 0),
            lambda: self.assign(
                scores, decode_bboxes, anchors, gt_labels, gt_bboxes, gt_mask
            ),
            lambda: (
                tf.zeros_like(decode_bboxes),
                tf.zeros_like(scores),
                tf.zeros_like(scores[..., 0]),
            ),
        )

    def count_params(self):
        return 0
