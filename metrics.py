import tensorflow as tf
import keras
# import tensorflow.keras as keras
from loss import maximum, minimum


def compute_ious(boxes1, boxes2, classes1, classes2):
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

    mask = (classes1 == classes2)

    return iou * mask


class MeanAveragePrecision(keras.metrics.Metric):
    def __init__(self, num_classes, *args, **kwargs):
        super(MeanAveragePrecision, self).__init__(name='mAP', *args, **kwargs)
        self.num_classes = num_classes

    def update_state(
            self,
            y,
            y_pred,
            sample_weight,
            *args,
            **kwargs
    ):
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        actual_positives = 0
        for i in range(y_pred['boxes'].shape[0]):
            actual_positives += y['num_objects'][i]
            for c in range(self.num_classes):
                for k in range(y_pred['num_detections'][i]):
                    if c != y_pred['classes'][i, k]:
                        continue
                    ious = compute_ious(
                        tf.expand_dims(y_pred['boxes'][i, k], axis=0),
                        y['boxes'][i],
                        tf.expand_dims(y_pred['classes'][i, k], axis=0),
                        y['classes'][i]
                    )
                    idx = tf.argmax(ious)
                    for j, t in enumerate(thresholds):
                        if ious[idx] > t:
                            self.positives[j].append((y_pred['confidence'][i, k], True))
                        else:
                            self.positives[j].append((y_pred['confidence'][i, k], False))

    def result(self):
        pass
