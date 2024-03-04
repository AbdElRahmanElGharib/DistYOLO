import keras.layers
from model import *
from prediction_decoder import *
from keras_cv.src.models.object_detection.yolo_v8.yolo_v8_label_encoder import YOLOV8LabelEncoder
from keras_cv.src.losses.ciou_loss import CIoULoss


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

    def predict_step(self, *args):
        outputs = super(YOLO, self).predict_step(*args)

        return self.prediction_decoder({'preds': outputs, 'images': args[-1]})
