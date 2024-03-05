import tensorflow as tf
import keras
# import tensorflow.keras as keras
import math


def maximum(x1, x2):
    if isinstance(x1, tf.SparseTensor):
        if isinstance(x2, tf.SparseTensor):
            return tf.sparse.maximum(x1, x2)
        else:
            x1 = tf.sparse.to_dense(x1)
    elif isinstance(x2, tf.SparseTensor):
        x2 = tf.sparse.to_dense(x2)
    return tf.math.maximum(x1, x2)


def minimum(x1, x2):
    if isinstance(x1, tf.SparseTensor):
        if isinstance(x2, tf.SparseTensor):
            return tf.sparse.minimum(x1, x2)
        else:
            x1 = tf.sparse.to_dense(x1)
    elif isinstance(x2, tf.SparseTensor):
        x2 = tf.sparse.to_dense(x2)
    return tf.math.minimum(x1, x2)


def compute_ciou(boxes1, boxes2):
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxes1[..., :4], 4, axis=-1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxes2[..., :4], 4, axis=-1)

    width_1 = x_max1 - x_min1
    height_1 = y_max1 - y_min1 + keras.backend.epsilon()
    width_2 = x_max2 - x_min2
    height_2 = y_max2 - y_min2 + keras.backend.epsilon()

    intersection_area = maximum(
        minimum(x_max1, x_max2) - maximum(x_min1, x_min2), 0
    ) * maximum(
        minimum(y_max1, y_max2) - maximum(y_min1, y_min2), 0
    )
    union_area = (
        width_1 * height_1
        + width_2 * height_2
        - intersection_area
        + keras.backend.epsilon()
    )
    iou = tf.squeeze(
        tf.divide(intersection_area, union_area + keras.backend.epsilon()),
        axis=-1,
    )

    convex_width = maximum(x_max1, x_max2) - minimum(x_min1, x_min2)
    convex_height = maximum(y_max1, y_max2) - minimum(y_min1, y_min2)
    convex_diagonal_squared = tf.squeeze(
        convex_width**2 + convex_height**2 + keras.backend.epsilon(),
        axis=-1,
    )
    centers_distance_squared = tf.squeeze(
        ((x_min1 + x_max1) / 2 - (x_min2 + x_max2) / 2) ** 2
        + ((y_min1 + y_max1) / 2 - (y_min2 + y_max2) / 2) ** 2,
        axis=-1,
    )

    v = tf.squeeze(
        tf.pow(
            (4 / math.pi**2)
            * (tf.math.atan(width_2 / height_2) - tf.atan(width_1 / height_1)),
            2,
        ),
        axis=-1,
    )
    alpha = v / (v - iou + (1 + keras.backend.epsilon()))

    return iou - (
        centers_distance_squared / convex_diagonal_squared + v * alpha
    )


class CIoULoss(keras.losses.Loss):
    def __init__(self, eps=1e-7, **kwargs):
        super(CIoULoss, self).__init__(**kwargs)
        self.eps = eps

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        if y_pred.shape[-1] != 4:
            raise ValueError(
                "CIoULoss expects y_pred.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_pred.shape[-1]={y_pred.shape[-1]}."
            )

        if y_true.shape[-1] != 4:
            raise ValueError(
                "CIoULoss expects y_true.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_true.shape[-1]={y_true.shape[-1]}."
            )

        if y_true.shape[-2] != y_pred.shape[-2]:
            raise ValueError(
                "CIoULoss expects number of boxes in y_pred to be equal to the "
                "number of boxes in y_true. Received number of boxes in "
                f"y_true={y_true.shape[-2]} and number of boxes in "
                f"y_pred={y_pred.shape[-2]}."
            )

        ciou = compute_ciou(y_true, y_pred)
        return 1 - ciou
