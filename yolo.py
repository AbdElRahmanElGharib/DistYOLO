import keras.layers
from model import *
from prediction_decoder import *
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_label_encoder import YOLOV8LabelEncoder
from loss import CIoULoss


def maximum(x1, x2):
    if isinstance(x1, tf.SparseTensor):
        if isinstance(x2, tf.SparseTensor):
            return tf.sparse.maximum(x1, x2)
        else:
            x1 = tf.sparse.to_dense(x1)
    elif isinstance(x2, tf.SparseTensor):
        x2 = tf.sparse.to_dense(x2)
    return tf.math.maximum(x1, x2)


class YOLO(keras.Model):
    def __init__(self, num_classes=20, depth=1.0, width=1.0, ratio=1.0, *args, **kwargs):
        super(YOLO, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.feature_extractor = FeatureExtractor(depth, width, ratio)
        self.fpn = FPN(depth, width, ratio)
        self.detection_head = DetectionHead(num_classes, width)
        self.prediction_decoder = PredictionDecoder()
        self.classification_loss = keras.losses.BinaryCrossentropy(reduction="sum")
        self.box_loss = CIoULoss(bounding_box_format="xyxy", reduction="sum")
        self.label_encoder = YOLOV8LabelEncoder(num_classes=num_classes)
        self.box_loss_weight = 7.5
        self.classification_loss_weight = 0.5
        self.build((None, 640, 640, 3))

    def compile(
        self,
        box_loss_weight=7.5,
        classification_loss_weight=0.5,
        **kwargs,
    ):
        self.box_loss_weight = box_loss_weight
        self.classification_loss_weight = classification_loss_weight

        losses = {
            "box": self.box_loss,
            "class": self.classification_loss,
        }

        super(YOLO, self).compile(loss=losses, **kwargs)

    def call(self, inputs, training=None, mask=None):
        x = tf.image.resize(inputs, (640, 640))
        x = self.feature_extractor(x)
        x = self.fpn(x)
        x = self.detection_head(x)
        return x

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        pred_boxes = y_pred['boxes']
        pred_scores = y_pred['classes']

        pred_boxes = tf.reshape(pred_boxes, shape=(-1, 4, 16))
        pred_boxes = tf.nn.softmax(logits=pred_boxes, axis=-1) * tf.range(16, dtype='float32')
        pred_boxes = tf.math.reduce_sum(pred_boxes, axis=-1)

        anchor_points, stride_tensor = get_anchors(image_shape=x.shape[1:])
        stride_tensor = tf.expand_dims(stride_tensor, axis=-1)

        true_labels = y["classes"]
        true_boxes = y["boxes"]

        mask_gt = tf.math.reduce_all(true_boxes > -1.0, axis=-1, keepdims=True)

        pred_boxes = dist2bbox(pred_boxes, anchor_points)

        target_boxes, target_scores, fg_mask = self.label_encoder(
            pred_scores,
            tf.expand_dims(tf.cast(pred_boxes * stride_tensor, true_boxes.dtype), axis=0),
            anchor_points * stride_tensor,
            true_labels,
            true_boxes,
            mask_gt,
        )

        target_boxes /= stride_tensor
        target_scores_sum = maximum(tf.math.reduce_sum(target_scores), 1)
        box_weight = tf.expand_dims(
            tf.math.reduce_sum(target_scores, axis=-1) * fg_mask,
            axis=-1,
        )

        y_true = {
            "box": target_boxes * fg_mask[..., None],
            "class": target_scores,
        }
        y_pred = {
            "box": pred_boxes * fg_mask[..., None],
            "class": pred_scores,
        }
        sample_weights = {
            "box": self.box_loss_weight * box_weight / target_scores_sum,
            "class": self.classification_loss_weight / target_scores_sum,
        }

        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )

    def predict_step(self, *args):
        outputs = super(YOLO, self).predict_step(*args)

        return self.prediction_decoder({'preds': outputs, 'images': args[-1]})