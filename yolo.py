import tensorflow as tf
import keras
# import tensorflow.keras as keras
from model import FeatureExtractor, FPN, DetectionHead
from prediction_decoder import PredictionDecoder, get_anchors, dist2bbox
from label_encoder import LabelEncoder
from loss import CIoULoss, maximum


class YOLO(keras.Model):
    def __init__(
            self,
            target_dist,
            num_classes=20,
            depth=1.0,
            width=1.0,
            ratio=1.0,
            conf_threshold=0.2,
            iou_threshold=0.7,
            *args,
            **kwargs
    ):
        super(YOLO, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.feature_extractor = FeatureExtractor(depth, width, ratio)
        self.fpn = FPN(depth, width, ratio)
        self.detection_head = DetectionHead(num_classes, width)
        self.prediction_decoder = PredictionDecoder(conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        self.classification_loss = keras.losses.BinaryCrossentropy(reduction="sum")
        self.box_loss = CIoULoss(reduction="sum")
        self.distance_loss = keras.losses.BinaryCrossentropy(reduction="sum")
        self.label_encoder = LabelEncoder(num_classes=num_classes)
        self.box_loss_weight = None
        self.classification_loss_weight = None
        self.distance_loss_weight = None
        self.max_dist = tf.reduce_max(target_dist)
        self.build((None, 640, 640, 3))

    def compile(
        self,
        box_loss_weight=4.5,
        classification_loss_weight=0.45,
        distance_loss_weight=0.05,
        **kwargs,
    ):

        self.box_loss_weight = box_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.distance_loss_weight = distance_loss_weight

        losses = {
            "box": self.box_loss,
            "class": self.classification_loss,
            "distance": self.distance_loss
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
        pred_dist = y_pred['distances']

        pred_boxes = tf.reshape(pred_boxes, shape=(-1, 8400, 4, 16))
        pred_boxes = tf.nn.softmax(logits=pred_boxes, axis=-1) * tf.range(16, dtype='float32')
        pred_boxes = tf.math.reduce_sum(pred_boxes, axis=-1)

        anchor_points, stride_tensor = get_anchors(image_shape=x.shape[1:])
        stride_tensor = tf.expand_dims(stride_tensor, axis=-1)

        true_labels = y["classes"]
        true_boxes = y["boxes"]
        true_dist = y["distances"]

        mask_gt = tf.math.reduce_all(true_boxes > -1.0, axis=-1, keepdims=True)

        pred_boxes = dist2bbox(pred_boxes, tf.expand_dims(anchor_points, axis=0))

        target_boxes, target_scores, target_dist, fg_mask = self.label_encoder(
            {
                'scores': pred_scores,
                'decode_bboxes': tf.cast(pred_boxes * stride_tensor, true_boxes.dtype),
                'distances': pred_dist,
                'anchors': anchor_points * stride_tensor,
                'gt_labels': true_labels,
                'gt_bboxes': true_boxes,
                'gt_distances': true_dist,
                'gt_mask': mask_gt
            }
        )

        pred_dist = tf.where(
            target_dist == -1.,
            -1.,
            pred_dist
        )

        target_boxes /= stride_tensor
        target_scores_sum = maximum(tf.math.reduce_sum(target_scores), 1)
        box_weight = tf.expand_dims(
            tf.math.reduce_sum(target_scores, axis=-1) * fg_mask,
            axis=-1,
        )

        target_dist /= self.max_dist

        y_true = {
            "box": target_boxes * fg_mask[..., None],
            "class": target_scores,
            "distance": target_dist
        }
        y_pred = {
            "box": pred_boxes * fg_mask[..., None],
            "class": pred_scores,
            "distance": pred_dist
        }
        sample_weights = {
            "box": self.box_loss_weight * box_weight / target_scores_sum,
            "class": self.classification_loss_weight / target_scores_sum,
            "distance": self.distance_loss_weight / target_scores_sum
        }

        return super().compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )

    def predict_step(self, *args):
        outputs = super(YOLO, self).predict_step(*args)

        return self.prediction_decoder({'preds': outputs, 'images': args[-1], 'max_dist': self.max_dist})
