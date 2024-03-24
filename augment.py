import tensorflow as tf
import keras
# import tensorflow.keras as keras


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
        boxes = inputs['bounding_boxes']['boxes']

        prop = tf.random.uniform([])
        if prop <= self.rate:

            if self.horizontal:
                images = tf.image.flip_left_right(images)
                boxes = tf.map_fn(
                    lambda x: tf.map_fn(
                        lambda y: tf.constant([640 - y[0], y[1], 640 - y[2], y[3]]),
                        x
                    ),
                    boxes
                )

            if self.vertical:
                images = tf.image.flip_up_down(images)
                boxes = tf.map_fn(
                    lambda x: tf.map_fn(
                        lambda y: tf.constant([y[0], 640 - y[1], y[2], 640 - y[3]]),
                        x
                    ),
                    boxes
                )

        return {
            'images': images,
            'bounding_boxes': {
                'boxes': boxes,
                'distances': inputs['bounding_boxes']['distances'],
                'classes': inputs['bounding_boxes']['classes']
            }
        }
