import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Add, Resizing, Reshape, MaxPool2D, Concatenate, UpSampling2D, Input
from keras.losses import MeanSquaredError, BinaryFocalCrossentropy
from prediction_decoder import PredictionDecoder, get_anchors, dist2bbox
from label_encoder import LabelEncoder
from loss import maximum, CIoULoss

BOX_REGRESSORS = 64


class MobileYOLOv3(Model):
    def __init__(
            self,
            num_classes=80,
            segmentation_classes=1,
            conf_threshold=0.2,
            iou_threshold=0.7,
            *args,
            **kwargs
    ):
        super(MobileYOLOv3, self).__init__(*args, **kwargs)

        input_image = Input(shape=(None, None, 3), name='input')

        x = Resizing(320, 320)(input_image)

        x = Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)

        def bottleneck(x_in, in_channels, out_channels, expansion_factor, stride, nl):
            mid_channels = in_channels * expansion_factor

            _x = Conv2D(mid_channels, kernel_size=1, use_bias=False)(x_in)
            _x = BatchNormalization()(_x)
            _x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(_x)
            _x = BatchNormalization()(_x)

            _x = Conv2D(out_channels, kernel_size=1, use_bias=False)(_x)
            _x = BatchNormalization()(_x)
            _x = Activation('relu' if nl == 'RE' else 'swish')(_x)

            if in_channels == out_channels and stride == 1:
                _x = Add()([_x, x_in])

            return _x

        x = bottleneck(x, 16, 24, 2, 4, nl='RE')
        x1 = bottleneck(x, 24, 24, 3, 1, nl='RE')

        x = bottleneck(x1, 24, 32, 2, 2, nl='HS')
        x2 = bottleneck(x, 32, 48, 3, 1, nl='HS')

        x = bottleneck(x2, 48, 48, 2, 2, nl='HS')
        x3 = bottleneck(x, 48, 64, 3, 1, nl='HS')

        sppf1 = MaxPool2D(5, 1, padding='same')(x3)
        sppf2 = MaxPool2D(9, 1, padding='same')(x3)
        sppf3 = MaxPool2D(13, 1, padding='same')(x3)
        sppf4 = Conv2D(64, 1)(x3)

        sppf = Concatenate()([sppf1, sppf2, sppf3, sppf4])

        x = Conv2D(64, 1, use_bias=False)(sppf)
        x = BatchNormalization()(x)
        x_mid = Activation('leaky_relu')(x)

        x = Conv2D((num_classes + BOX_REGRESSORS + 1), 1, use_bias=False)(x_mid)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)

        out_1 = Reshape((-1, (num_classes + BOX_REGRESSORS + 1)))(x)

        x = Conv2D(48, 1, use_bias=False)(x_mid)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)
        x = UpSampling2D(size=2)(x)

        x = Concatenate()([x, x2])

        x = Conv2D(48, 1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x_mid = Activation('leaky_relu')(x)

        x = Conv2D((num_classes + BOX_REGRESSORS + 1), 1, use_bias=False)(x_mid)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)

        out_2 = Reshape((-1, (num_classes + BOX_REGRESSORS + 1)))(x)

        x_detections = Concatenate(axis=-2)([out_1, out_2])

        detections = Activation('linear', name='detections')(x_detections)

        x = Conv2D(filters=32, kernel_size=1, use_bias=False)(x3)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)
        x = UpSampling2D(size=4)(x)
        x = Conv2D(filters=32, kernel_size=1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)
        x = Conv2D(filters=16, kernel_size=1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)
        x = Conv2D(filters=segmentation_classes, kernel_size=1)(x)
        segments = Activation('sigmoid', name='segments')(x)

        self.model = Model(inputs=[input_image], outputs=[detections, segments], name='mobile_yolo')

        self.prediction_decoder = PredictionDecoder(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            anchor_strides=(16, 32)
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

        self.build((None, None, None, 3))

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

        super(MobileYOLOv3, self).compile(loss=losses, **kwargs)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        true_labels, true_boxes, true_dist, true_mask, training_mode = y
        detections, pred_mask = y_pred
        pred_boxes = detections[..., :64]
        pred_scores = detections[..., 64:-1]
        pred_dist = detections[..., -1:]

        pred_boxes = tf.reshape(pred_boxes, shape=(-1, 500, 4, 16))
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

        return super(MobileYOLOv3, self).compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs, training, mask)

    def predict_step(self, *args):
        outputs = super(MobileYOLOv3, self).predict_step(*args)

        def reformat(x_in):
            return {
                'boxes': x_in[0][..., :64],
                'classes': x_in[0][..., 64:-1],
                'distances': x_in[0][..., -1:]
            }

        return {
            **self.prediction_decoder({'preds': reformat(outputs), 'images': args[-1]}),
            'segments': outputs[1]
        }

    def train_step(self, data):
        if not isinstance(data, tuple):
            data = tuple(data)

        x, y = data[0], data[1]
        sample_weight = None

        if len(data) == 3:
            sample_weight = data[2]

        return super(MobileYOLOv3, self).train_step((x, y, sample_weight))


if __name__ == '__main__':
    model = MobileYOLOv3()
    model.summary()
    # 100k params
    # 0.4 GFLOPS
    # 250 fps on "Processor	Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz, 2592 Mhz, 6 Core(s), 12 Logical Processor(s)" using ONNX Framework
