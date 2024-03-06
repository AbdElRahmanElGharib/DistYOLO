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
