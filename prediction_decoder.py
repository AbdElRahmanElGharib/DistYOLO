import tensorflow as tf
import keras
# import tensorflow.keras as keras
from keras_cv.src import bounding_box, layers


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
