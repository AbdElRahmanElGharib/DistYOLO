import tensorflow as tf
import keras
# import tensorflow.keras as keras
from loss import maximum, compute_ciou


def convert_bounding_box_to_dense(bounding_boxes):
    if isinstance(bounding_boxes["classes"], tf.RaggedTensor):
        bounding_boxes["classes"] = bounding_boxes["classes"].to_tensor(
            default_value=-1,
            shape=None
        )

    if isinstance(bounding_boxes["distances"], tf.RaggedTensor):
        bounding_boxes["distances"] = bounding_boxes["distances"].to_tensor(
            default_value=-1,
            shape=None
        )

    if isinstance(bounding_boxes["boxes"], tf.RaggedTensor):
        shape = list(bounding_boxes["boxes"].shape)
        shape[-1] = 4
        bounding_boxes["boxes"] = bounding_boxes["boxes"].to_tensor(
            default_value=-1,
            shape=shape
        )
    return bounding_boxes


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

    def assign(self, scores, decode_bboxes, distances, anchors, gt_labels, gt_bboxes, gt_distances, gt_mask):
        num_anchors = anchors.shape[0]
        # TODO: use tf.gather_nd instead of tf.experimental.numpy.take_along_axis
        bbox_scores = tf.experimental.numpy.take_along_axis(
            scores,
            tf.cast(maximum(gt_labels[:, None, :], 0), "int32"),
            axis=-1,
        )
        bbox_scores = tf.transpose(
            bbox_scores,
            (0, 2, 1)
        )

        overlaps = compute_ciou(
            tf.expand_dims(gt_bboxes, axis=2),
            tf.expand_dims(decode_bboxes, axis=1)
        )

        alignment_metrics = tf.pow(bbox_scores, self.alpha) * tf.pow(overlaps, self.beta)
        alignment_metrics = tf.where(
            gt_mask,
            alignment_metrics,
            0
        )

        matching_anchors_in_gt_boxes = is_anchor_center_within_box(
            anchors,
            gt_bboxes
        )

        alignment_metrics = tf.where(
            matching_anchors_in_gt_boxes,
            alignment_metrics,
            0
        )

        candidate_metrics, candidate_idxs = tf.math.top_k(
            alignment_metrics,
            self.max_anchor_matches,
            sorted=True
        )
        candidate_idxs = tf.where(candidate_metrics > 0, candidate_idxs, -1)

        anchors_matched_gt_box = tf.zeros_like(overlaps)
        for k in range(self.max_anchor_matches):
            anchors_matched_gt_box += tf.one_hot(
                candidate_idxs[:, :, k],
                num_anchors,
                axis=-1
            )

        overlaps *= anchors_matched_gt_box

        gt_box_matches_per_anchor = tf.argmax(
            overlaps,
            axis=1
        )
        gt_box_matches_per_anchor_mask = tf.math.reduce_max(overlaps, axis=1) > 0

        gt_box_matches_per_anchor = tf.cast(gt_box_matches_per_anchor, "int32")
        # TODO: use tf.gather_nd instead of tf.experimental.numpy.take_along_axis
        bbox_labels = tf.experimental.numpy.take_along_axis(
            gt_bboxes,
            gt_box_matches_per_anchor[:, :, None],
            axis=1
        )
        bbox_labels = tf.where(
            gt_box_matches_per_anchor_mask[:, :, None],
            bbox_labels,
            -1
        )
        # TODO: use tf.gather_nd instead of tf.experimental.numpy.take_along_axis
        class_labels = tf.experimental.numpy.take_along_axis(
            gt_labels,
            gt_box_matches_per_anchor,
            axis=1
        )
        class_labels = tf.where(
            gt_box_matches_per_anchor_mask,
            class_labels,
            -1
        )
        # TODO: use tf.gather_nd instead of tf.experimental.numpy.take_along_axis
        dist_labels = tf.experimental.numpy.take_along_axis(
            gt_distances,
            gt_box_matches_per_anchor,
            axis=1
        )
        dist_labels = tf.where(
            gt_box_matches_per_anchor_mask,
            dist_labels,
            -1
        )

        class_labels = tf.one_hot(
            tf.cast(class_labels, "int32"),
            self.num_classes,
            axis=-1
        )

        alignment_metrics *= anchors_matched_gt_box
        max_alignment_per_gt_box = tf.math.reduce_max(
            alignment_metrics,
            axis=-1,
            keepdims=True
        )
        max_overlap_per_gt_box = tf.math.reduce_max(
            overlaps,
            axis=-1,
            keepdims=True
        )

        normalized_alignment_metrics = tf.math.reduce_max(
            alignment_metrics
            * max_overlap_per_gt_box
            / (max_alignment_per_gt_box + self.epsilon),
            axis=-2,
        )

        class_labels *= normalized_alignment_metrics[:, :, None]
        dist_labels *= normalized_alignment_metrics[:, :, None]

        bbox_labels = tf.reshape(
            bbox_labels,
            (-1, num_anchors, 4)
        )
        return (
            tf.stop_gradient(bbox_labels),
            tf.stop_gradient(class_labels),
            tf.stop_gradient(dist_labels),
            tf.stop_gradient(
                tf.cast(gt_box_matches_per_anchor > -1, "float32")
            ),
        )

    def call(self, inputs, *args, **kwargs):
        scores = inputs['scores']
        decode_bboxes = inputs['decode_bboxes']
        distances = inputs['distances']
        anchors = inputs['anchors']
        gt_labels = inputs['gt_labels']
        gt_bboxes = inputs['gt_bboxes']
        gt_mask = inputs['gt_mask']
        gt_distances = inputs['gt_distances']

        if isinstance(gt_bboxes, tf.RaggedTensor):
            dense_bounding_boxes = convert_bounding_box_to_dense(
                {
                    "boxes": gt_bboxes,
                    "classes": gt_labels,
                    "distances": gt_distances
                }
            )
            gt_bboxes = dense_bounding_boxes["boxes"]
            gt_labels = dense_bounding_boxes["classes"]
            gt_distances = dense_bounding_boxes["distances"]

        if isinstance(gt_mask, tf.RaggedTensor):
            gt_mask = gt_mask.to_tensor()

        max_num_boxes = tf.shape(gt_bboxes)[1]

        return tf.cond(
            max_num_boxes > 0,
            lambda: self.assign(
                scores, decode_bboxes, distances, anchors, gt_labels, gt_bboxes, gt_distances, gt_mask
            ),
            lambda: (
                tf.zeros_like(decode_bboxes),
                tf.zeros_like(scores),
                tf.zeros_like(distances),
                tf.zeros_like(scores[..., 0]),
            ),
        )

    def count_params(self):
        return 0
