import tensorflow as tf
import keras
# import tensorflow.keras as keras
from non_max_suppression import NonMaxSuppression


def get_anchors(
    image_shape,
    strides=(8, 16, 32),
    base_anchors=(0.5, 0.5),
):
    base_anchors = tf.constant(base_anchors, dtype="float32")

    all_anchors = []
    all_strides = []
    for stride in strides:
        hh_centers = tf.range(0, image_shape[0], stride)
        ww_centers = tf.range(0, image_shape[1], stride)
        ww_grid, hh_grid = tf.meshgrid(ww_centers, hh_centers)
        grid = tf.cast(
            tf.reshape(tf.stack([hh_grid, ww_grid], 2), [-1, 1, 2]),
            "float32",
        )
        anchors = (
            tf.expand_dims(
                base_anchors * tf.constant([stride, stride], "float32"), 0
            )
            + grid
        )
        anchors = tf.reshape(anchors, [-1, 2])
        all_anchors.append(anchors)
        all_strides.append(tf.repeat(stride, anchors.shape[0]))

    all_anchors = tf.cast(tf.concat(all_anchors, axis=0), "float32")
    all_strides = tf.cast(tf.concat(all_strides, axis=0), "float32")

    all_anchors = all_anchors / all_strides[:, None]

    # Swap the x and y coordinates of the anchors.
    all_anchors = tf.concat(
        [all_anchors[:, 1, None], all_anchors[:, 0, None]], axis=-1
    )
    return all_anchors, all_strides


def dist2bbox(distance, anchor_points):
    left_top, right_bottom = tf.split(distance, 2, axis=-1)
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    return tf.concat((x1y1, x2y2), axis=-1)  # xyxy bbox


class PredictionDecoder(keras.Model):
    def __init__(self, conf_threshold=0.2, iou_threshold=0.7, anchor_strides=(8, 16, 32), image_shape=None, *args, **kwargs):
        super(PredictionDecoder, self).__init__(*args, **kwargs)
        self.boxes_reshape = keras.layers.Reshape(target_shape=(-1, 4, 16))
        self.anchor_strides = anchor_strides
        self.nms = NonMaxSuppression(
                confidence_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
        self.image_shape = image_shape

    def call(self, inputs, training=None, mask=None):
        preds, images = inputs['preds'], inputs['images']

        if self.image_shape is None:
            self.image_shape = images.shape[1:]

        boxes = preds['boxes']
        scores = preds['classes']
        distances = preds['distances']

        boxes = self.boxes_reshape(boxes)
        boxes = tf.nn.softmax(logits=boxes, axis=-1) * tf.range(16, dtype='float32')
        boxes = tf.math.reduce_sum(boxes, axis=-1)

        anchor_points, stride_tensor = get_anchors(image_shape=self.image_shape, strides=self.anchor_strides)
        stride_tensor = tf.expand_dims(stride_tensor, axis=-1)
        box_preds = dist2bbox(boxes, anchor_points) * stride_tensor

        return self.nms({'boxes': box_preds, 'classes': scores, 'distances': distances})

    def count_params(self):
        return 0
