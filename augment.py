import tensorflow as tf
import keras
# import tensorflow.keras as keras
from label_encoder import convert_bounding_box_to_dense


HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


# keras.layers.Layer.
class RandomFlip(keras.layers.Layer):
    def __init__(self, mode=HORIZONTAL, rate=0.5, **kwargs):
        self.horizontal = False
        self.vertical = False

        self.rate = rate

        if mode == HORIZONTAL:
            self.horizontal = True
            self.vertical = False

        elif mode == VERTICAL:
            self.horizontal = False
            self.vertical = True

        elif mode == HORIZONTAL_AND_VERTICAL:
            self.horizontal = True
            self.vertical = True

        else:
            raise ValueError(
                "RandomFlip layer {name} received an unknown mode="
                "{arg}".format(name=self.name, arg=mode)
            )

        super(RandomFlip, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        images = inputs['images']
        bounding_boxes = inputs['bounding_boxes']

        bounding_boxes = convert_bounding_box_to_dense(bounding_boxes)

        if isinstance(images, tf.RaggedTensor):
            images = images.to_tensor(
                default_value=-1,
                shape=None
            )

        boxes = bounding_boxes['boxes']

        prop = tf.random.uniform([])
        if prop <= self.rate:

            if self.horizontal:
                images = tf.map_fn(
                    tf.image.flip_left_right,
                    images
                )

                boxes *= tf.constant([[[-1, 1, -1, 1]]])
                boxes += tf.constant([[[640, 0, 640, 0]]])
                boxes = tf.concat(
                    [
                        tf.expand_dims(boxes[..., 2], axis=-1),
                        tf.expand_dims(boxes[..., 1], axis=-1),
                        tf.expand_dims(boxes[..., 0], axis=-1),
                        tf.expand_dims(boxes[..., 3], axis=-1)
                    ],
                    axis=-1
                )

            if self.vertical:
                images = tf.map_fn(
                    tf.image.flip_up_down,
                    images
                )

                boxes *= tf.constant([[[1, -1, 1, -1]]])
                boxes += tf.constant([[[0, 640, 0, 640]]])
                boxes = tf.concat(
                    [
                        tf.expand_dims(boxes[..., 0], axis=-1),
                        tf.expand_dims(boxes[..., 3], axis=-1),
                        tf.expand_dims(boxes[..., 2], axis=-1),
                        tf.expand_dims(boxes[..., 1], axis=-1)
                    ],
                    axis=-1
                )

        return {
            'images': images,
            'bounding_boxes': {
                'boxes': boxes,
                'distances': inputs['bounding_boxes']['distances'],
                'classes': inputs['bounding_boxes']['classes']
            }
        }
