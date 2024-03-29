import tensorflow as tf
import keras
# import tensorflow.keras as keras
from label_encoder import convert_bounding_box_to_dense


HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


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

        if rate < 0.0 or rate > 1.0:
            raise ValueError(
                f"`rate` should be inside of range [0, 1]. Got rate={rate}"
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

                boxes *= tf.constant([[[-1, 1, -1, 1]]], dtype=boxes.dtype)
                boxes += tf.constant([[[640, 0, 640, 0]]], dtype=boxes.dtype)
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

                boxes *= tf.constant([[[1, -1, 1, -1]]], dtype=boxes.dtype)
                boxes += tf.constant([[[0, 640, 0, 640]]], dtype=boxes.dtype)
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

    def count_params(self):
        return 0


class ChannelShuffle(keras.layers.Layer):
    def __init__(self, rate=0.5, **kwargs):
        self.rate = rate

        if rate < 0.0 or rate > 1.0:
            raise ValueError(
                f"`rate` should be inside of range [0, 1]. Got rate={rate}"
            )

        super(ChannelShuffle, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        images = inputs['images']

        if isinstance(images, tf.RaggedTensor):
            images = images.to_tensor(
                default_value=-1,
                shape=None
            )

        prop = tf.random.uniform([])
        if prop <= self.rate:
            indices = tf.random.shuffle(tf.constant([0, 1, 2]))

            images = tf.concat(
                [
                    tf.expand_dims(images[..., indices[0]], axis=-1),
                    tf.expand_dims(images[..., indices[1]], axis=-1),
                    tf.expand_dims(images[..., indices[2]], axis=-1),
                ],
                axis=-1
            )

        return {
            'images': images,
            'bounding_boxes': inputs['bounding_boxes']
        }

    def count_params(self):
        return 0


class RandomHue(keras.layers.Layer):
    def __init__(self, rate=0.5, limit=0.5, deterministic=False, **kwargs):
        self.rate = rate
        self.limit = limit
        self.deterministic = deterministic

        if rate < 0.0 or rate > 1.0:
            raise ValueError(
                f"`rate` should be inside of range [0, 1]. Got rate={rate}"
            )

        if limit < 0.0 or limit > 1.0:
            raise ValueError(
                f"`limit` should be inside of range [0, 1]. Got limit={limit}"
            )

        super(RandomHue, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        images = inputs['images']

        if isinstance(images, tf.RaggedTensor):
            images = images.to_tensor(
                default_value=-1,
                shape=None
            )

        prop = tf.random.uniform([])
        if prop <= self.rate:
            delta = tf.random.uniform(
                shape=[],
                minval=-self.limit,
                maxval=self.limit
            )

            if self.deterministic:
                delta = self.limit

            images = tf.image.adjust_hue(images, delta)

        return {
            'images': images,
            'bounding_boxes': inputs['bounding_boxes']
        }

    def count_params(self):
        return 0


class RandomSaturation(keras.layers.Layer):
    def __init__(self, rate=0.5, limit=2.0, deterministic=False, **kwargs):
        self.rate = rate
        self.limit = limit
        self.deterministic = deterministic

        if rate < 0.0 or rate > 1.0:
            raise ValueError(
                f"`rate` should be inside of range [0, 1]. Got rate={rate}"
            )

        if limit < 1.0:
            raise ValueError(
                f"`limit` should be inside of range [1, inf]. Got limit={limit}"
            )

        super(RandomSaturation, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        images = inputs['images']

        if isinstance(images, tf.RaggedTensor):
            images = images.to_tensor(
                default_value=-1,
                shape=None
            )

        prop = tf.random.uniform([])
        if prop <= self.rate:
            delta = tf.random.uniform(
                shape=[],
                minval=1.0/self.limit,
                maxval=self.limit
            )

            if self.deterministic:
                delta = self.limit

            images = tf.image.adjust_saturation(images, delta)

        return {
            'images': images,
            'bounding_boxes': inputs['bounding_boxes']
        }

    def count_params(self):
        return 0


class RandomBrightness(keras.layers.Layer):
    def __init__(self, rate=0.5, limit=0.2, deterministic=False, **kwargs):
        self.rate = rate
        self.limit = int(limit*255)
        self.deterministic = deterministic

        if rate < 0.0 or rate > 1.0:
            raise ValueError(
                f"`rate` should be inside of range [0, 1]. Got rate={rate}"
            )

        if limit < 0.0 or limit > 1.0:
            raise ValueError(
                f"`limit` should be inside of range [0, 1]. Got limit={limit}"
            )

        super(RandomBrightness, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        images = inputs['images']

        if isinstance(images, tf.RaggedTensor):
            images = images.to_tensor(
                default_value=-1,
                shape=None
            )

        prop = tf.random.uniform([])
        if prop <= self.rate:
            delta = tf.random.uniform(
                shape=[],
                minval=-self.limit,
                maxval=self.limit
            )

            if self.deterministic:
                delta = self.limit

            delta = tf.cast(delta, dtype=images.dtype)

            images = tf.clip_by_value(
                images+delta,
                0, 255
            )

        return {
            'images': images,
            'bounding_boxes': inputs['bounding_boxes']
        }

    def count_params(self):
        return 0
