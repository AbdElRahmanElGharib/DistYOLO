import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Add, Resizing,\
    GlobalAveragePooling2D, Reshape, Multiply, MaxPool2D, Concatenate, UpSampling2D, Input
from keras.losses import MeanSquaredError, BinaryFocalCrossentropy
from prediction_decoder import PredictionDecoder, get_anchors, dist2bbox
from label_encoder import LabelEncoder
from loss import maximum, CIoULoss


BOX_REGRESSORS = 64


class MobileYOLOv2(Model):
    def __init__(
            self,
            num_classes=80,
            segmentation_classes=1,
            conf_threshold=0.2,
            iou_threshold=0.7,
            *args,
            **kwargs
    ):
        super(MobileYOLOv2, self).__init__(*args, **kwargs)
        input_image = Input(shape=(None, None, 3), name='input')
        x = Resizing(320, 320)(input_image)

        x = Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)

        def bottleneck(x_in, in_channels, out_channels, expansion_factor, stride, use_se, nl):
            mid_channels = in_channels * expansion_factor

            _x = Conv2D(mid_channels, kernel_size=1, use_bias=False)(x_in)
            _x = BatchNormalization()(_x)
            _x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(_x)
            _x = BatchNormalization()(_x)

            if use_se:
                x_se = GlobalAveragePooling2D()(_x)
                x_se = Reshape((1, 1, mid_channels))(x_se)
                x_se = Conv2D(mid_channels // 4, kernel_size=1, activation='relu', use_bias=False)(x_se)
                x_se = Conv2D(mid_channels, kernel_size=1, activation='sigmoid', use_bias=False)(x_se)
                _x = Multiply()([_x, x_se])

            _x = Conv2D(out_channels, kernel_size=1, use_bias=False)(_x)
            _x = BatchNormalization()(_x)
            _x = Activation('relu' if nl == 'RE' else 'swish')(_x)

            if in_channels == out_channels and stride == 1:
                _x = Add()([_x, x_in])

            return _x

        x = bottleneck(x, 16, 16, 1, 2, use_se=True, nl='RE')
        x = bottleneck(x, 16, 24, 2, 2, use_se=False, nl='RE')
        x1 = bottleneck(x, 24, 24, 3, 1, use_se=False, nl='RE')

        x = bottleneck(x1, 24, 40, 2, 2, use_se=True, nl='HS')
        x2 = bottleneck(x, 40, 48, 3, 1, use_se=True, nl='HS')

        x = bottleneck(x2, 48, 48, 3, 1, use_se=True, nl='HS')
        x = bottleneck(x, 48, 96, 6, 2, use_se=True, nl='HS')

        x = Conv2D(64, kernel_size=1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x3 = Activation('swish')(x)

        sppf1 = MaxPool2D(5, 1, padding='same')(x3)
        sppf2 = MaxPool2D(9, 1, padding='same')(x3)
        sppf3 = MaxPool2D(13, 1, padding='same')(x3)
        sppf4 = Conv2D(64, 1)(x3)

        sppf = Concatenate()([sppf1, sppf2, sppf3, sppf4])

        x = Conv2D(128, 1, use_bias=False)(sppf)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)

        x = bottleneck(x, 128, 128, 2, 1, False, 'RE')
        x_mid = bottleneck(x, 128, 128, 1, 1, False, 'RE')

        x = Conv2D((num_classes + BOX_REGRESSORS + 1), 1, use_bias=False)(x_mid)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)

        out_1 = Reshape((-1, (num_classes + BOX_REGRESSORS + 1)))(x)

        x = Conv2D(64, 1, use_bias=False)(x_mid)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)
        x = UpSampling2D(size=2)(x)

        x = Concatenate()([x, x2])

        x = Conv2D(64, 1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)

        x = bottleneck(x, 64, 64, 2, 1, False, 'RE')
        x_mid = bottleneck(x, 64, 64, 3, 1, False, 'RE')

        x = Conv2D((num_classes + BOX_REGRESSORS + 1), 1, use_bias=False)(x_mid)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)

        out_2 = Reshape((-1, (num_classes + BOX_REGRESSORS + 1)))(x)

        x = Conv2D(32, 1, use_bias=False)(x_mid)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)
        x = UpSampling2D(size=2)(x)

        x = Concatenate()([x, x1])

        x = Conv2D(32, 1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)

        x = bottleneck(x, 32, 32, 2, 1, False, 'RE')
        x = bottleneck(x, 32, 32, 3, 1, False, 'RE')

        x = Conv2D((num_classes + BOX_REGRESSORS + 1), 1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)

        out_3 = Reshape((-1, (num_classes + BOX_REGRESSORS + 1)))(x)

        x_detections = Concatenate(axis=-2)([out_1, out_2, out_3])

        detections = Activation('linear', name='detections')(x_detections)

        def conv(x_in, filters, kernel_size):
            _x = DepthwiseConv2D(kernel_size=kernel_size, padding='same', use_bias=False)(x_in)
            _x = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False)(_x)
            _x = BatchNormalization()(_x)
            _x = Activation('leaky_relu')(_x)
            return _x

        x = conv(x3, filters=64, kernel_size=3)

        x = UpSampling2D(size=4)(x)

        x = conv(x, filters=64, kernel_size=5)

        x = UpSampling2D(size=4)(x)

        x = conv(x, filters=64, kernel_size=5)
        x = UpSampling2D(size=2)(x)

        x = conv(x, filters=64, kernel_size=5)
        x = conv(x, filters=32, kernel_size=5)

        x = Conv2D(filters=segmentation_classes, kernel_size=1)(x)
        segments = Activation('sigmoid', name='segments')(x)
        
        self.model = Model(inputs=[input_image], outputs=[detections, segments], name='mobile_yolo')

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

        super(MobileYOLOv2, self).compile(loss=losses, **kwargs)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        true_labels, true_boxes, true_dist, true_mask, training_mode = y
        detections, pred_mask = y_pred
        pred_boxes = detections[..., :64]
        pred_scores = detections[..., 64:-1]
        pred_dist = detections[..., -1:]

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

        return super(MobileYOLOv2, self).compute_loss(
            x=x, y=y_true, y_pred=y_pred, sample_weight=sample_weights
        )

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs, training, mask)

    def predict_step(self, *args):
        outputs = super(MobileYOLOv2, self).predict_step(*args)

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

        return super(MobileYOLOv2, self).train_step((x, y, sample_weight))


if __name__ == '__main__':
    model = MobileYOLOv2()
    model.summary()

    import time
    import numpy as np

    times = []
    for i in range(100):
        t = time.time()
        model.predict(tf.constant(tf.ones((1, 320, 320, 3)), dtype=tf.float32))
        if i != 0:
            times.append(time.time() - t)
        print('iteration ', i + 1, ':\t', time.time() - t, 's')
    print('average time:\t', np.average(times))
