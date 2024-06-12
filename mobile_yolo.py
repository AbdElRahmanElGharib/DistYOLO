import tensorflow as tf
from tensorflow.keras import Model
from prediction_decoder import PredictionDecoder
from mobile_yolo_structure import MobileNetV3Backbone, DetectionHead, DistanceHead, SegmentationHead


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

        self.backbone = MobileNetV3Backbone(grouping_factor=grouping_factor)

        self.detection_head = DetectionHead(num_classes=num_classes, grouping_factor=grouping_factor)
        self.distance_head = DistanceHead(num_classes=num_classes, grouping_factor=grouping_factor)
        self.segmentation_head = SegmentationHead(num_classes=segmentation_classes, grouping_factor=grouping_factor)

        self.prediction_decoder = PredictionDecoder(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        self.build((None, 320, 320, 3))

    def call(self, inputs, training=None, mask=None):
        inputs = tf.image.resize(inputs, (320, 320))

        features = self.backbone(inputs, training=training)

        detections = self.detection_head(features, training=training)
        distances = self.distance_head((*features, detections), training=training)
        segments = self.segmentation_head((inputs, *features), training=training)

        return detections, distances, segments

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
