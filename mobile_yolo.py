import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Resizing, Activation
from tensorflow.keras.losses import MeanSquaredError, BinaryFocalCrossentropy
from mobile_yolo_structure import MobileNetV3Backbone, DetectionHead, DistanceHead, SegmentationHead
from prediction_decoder import PredictionDecoder, get_anchors, dist2bbox
from label_encoder import LabelEncoder
from loss import maximum, CIoULoss


class MobileYOLO(Model):
    def __init__(
            self,
            num_classes=80,
            segmentation_classes=1,
            grouping_factor=1,
            conf_threshold=0.2,
            iou_threshold=0.7,
            *args,
            **kwargs
    ):
        super(MobileYOLO, self).__init__(*args, **kwargs)

        self.image_resize = Resizing(320, 320)

        self.backbone = MobileNetV3Backbone(grouping_factor=grouping_factor)

        self.detection_head = DetectionHead(num_classes=num_classes, grouping_factor=grouping_factor)
        self.distance_head = DistanceHead(num_classes=num_classes, grouping_factor=grouping_factor)
        self.segmentation_head = SegmentationHead(num_classes=segmentation_classes, grouping_factor=grouping_factor)

        self.act_out_detections = Activation(activation='linear', name='detections')
        self.act_out_distances = Activation(activation='linear', name='distances')
        self.act_out_segments = Activation(activation='linear', name='segments')

        self.prediction_decoder = PredictionDecoder(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        self.label_encoder = LabelEncoder(num_classes=num_classes)

        self.box_loss = CIoULoss(reduction="sum")
        self.classification_loss = BinaryFocalCrossentropy(
                apply_class_balancing=True,
                alpha=0.25,
                gamma=2.0,
                reduction="sum"
            )
        self.distance_loss = MeanSquaredError(reduction="sum")
        self.segmentation_loss = BinaryFocalCrossentropy(
                apply_class_balancing=True,
                alpha=0.25,
                gamma=2.0,
                reduction="sum"
            )

        self.box_loss_weight = 7.5
        self.classification_loss_weight = 1.0
        self.distance_loss_weight = 0.5
        self.segmentation_loss_weight = 1.0

        self.build((None, 320, 320, 3))

    def compile(
            self,
            box_loss_weight=7.5,
            classification_loss_weight=1.5,
            distance_loss_weight=1.0,
            segmentation_loss_weight=1.0,
            **kwargs,
    ):
        self.box_loss_weight = box_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.segmentation_loss_weight = segmentation_loss_weight

        losses = {
            "box": self.box_loss,
            "class": self.classification_loss,
            "distance": self.distance_loss,
            "segmentation": self.segmentation_loss
        }

        super(MobileYOLO, self).compile(loss=losses, **kwargs)

    def call(self, inputs, training=None, mask=None):
        resized_inputs = self.image_resize(inputs)

        features = self.backbone(resized_inputs, training=training)

        detections = self.detection_head(features, training=training)
        distances = self.distance_head((*features, detections), training=training)
        segments = self.segmentation_head((resized_inputs, *features), training=training)

        detections = self.act_out_detections(detections)
        distances = self.act_out_distances(distances)
        segments = self.act_out_segments(segments)

        return detections, distances, segments

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        true_labels, true_boxes, true_dist, true_mask, training_mode = y
        detections, pred_dist, pred_mask = y_pred
        pred_boxes = detections[..., :64]
        pred_scores = detections[..., 64:]

        pred_boxes = tf.reshape(pred_boxes, shape=(-1, 8400, 4, 16))
        pred_boxes = tf.nn.softmax(logits=pred_boxes, axis=-1) * tf.range(16, dtype='float32')
        pred_boxes = tf.math.reduce_sum(pred_boxes, axis=-1)

        anchor_points, stride_tensor = get_anchors(image_shape=x.shape[1:])
        stride_tensor = tf.expand_dims(stride_tensor, axis=-1)

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

        target_boxes /= stride_tensor
        target_scores_sum = maximum(tf.math.reduce_sum(target_scores), 1)
        box_weight = tf.expand_dims(
            tf.math.reduce_sum(target_scores, axis=-1) * fg_mask,
            axis=-1,
        )
        target_boxes *= fg_mask[..., None]
        pred_boxes *= fg_mask[..., None]

        boxes_training_mask = tf.broadcast_to(training_mode == 0, pred_boxes.shape)
        classes_training_mask = tf.broadcast_to(training_mode == 0, pred_scores.shape)
        distance_training_mask = tf.broadcast_to(training_mode == 1, pred_dist.shape)
        segmentation_training_mask = tf.broadcast_to(training_mode == 2, pred_mask.shape)

        target_boxes = tf.where(boxes_training_mask, target_boxes, pred_boxes)
        target_scores = tf.where(classes_training_mask, target_scores, pred_scores)
        target_dist = tf.where(distance_training_mask, target_dist, pred_dist)
        true_mask = tf.where(segmentation_training_mask, true_mask, pred_mask)

        y_true = {
            "box": target_boxes,
            "class": target_scores,
            "distance": target_dist,
            "segmentation": true_mask
        }
        y_pred = {
            "box": pred_boxes,
            "class": pred_scores,
            "distance": pred_dist,
            "segmentation": pred_mask
        }
        sample_weights = {
            "box": 7.5 * box_weight / target_scores_sum,
            "class": 1.0 / target_scores_sum,
            "distance": 0.5 / target_scores_sum,
            "segmentation": 1.0
        }

        return super(MobileYOLO, self).compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )

    def predict_step(self, *args):
        outputs = super(MobileYOLO, self).predict_step(*args)

        def reformat(x_in):
            return {
                'boxes': x_in[0][..., :64],
                'classes': x_in[0][..., 64:],
                'distances': x_in[1]
            }

        return {
            **self.prediction_decoder({'preds': reformat(outputs), 'images': args[-1]}),
            'segments': outputs[2]
        }

    def train_step(self, data):
        if not isinstance(data, tuple):
            data = tuple(data)

        x, y = data[0], data[1]
        sample_weight = None

        if len(data) == 3:
            sample_weight = data[2]

        return super(MobileYOLO, self).train_step((x, y, sample_weight))
