import tensorflow as tf
import keras
# import tensorflow.keras as keras
# from keras.utils import get_custom_objects


# def ranged_leaky_relu(features, alpha=0.1, end=150.0):

#     mask = tf.cast(features < 0.0, features.dtype)

#     outputs = tf.where(mask == 0.0, features, features * alpha)

#     mask = tf.cast(features > end, features.dtype)

#     outputs = tf.where(mask == 0.0, outputs, outputs * alpha + end * (1-alpha))

#     return outputs


class Conv(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding,  *args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
        self.pad = None
        self.conv = None
        if isinstance(padding, str):
            self.pad = keras.layers.ZeroPadding2D(padding=0)
            self.conv = keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            padding=padding,
                                            use_bias=False)
        else:
            self.pad = keras.layers.ZeroPadding2D(padding=padding)
            self.conv = keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            use_bias=False)
        self.bn = keras.layers.BatchNormalization()
        self.act = keras.layers.Activation(activation=tf.nn.silu)

    def call(self, inputs, training=None, mask=None):
        x = self.pad(inputs)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SPPF(keras.Model):
    def __init__(self, channels, pool_size=5, *args, **kwargs):
        super(SPPF, self).__init__(*args, **kwargs)
        self.conv_in = Conv(channels//2, 1, 1, 'same')
        self.conv_out = Conv(channels, 1, 1, 'same')
        self.concat = keras.layers.Concatenate()
        self.pool1 = keras.layers.MaxPool2D(pool_size=pool_size,
                                            strides=1,
                                            padding='same')
        self.pool2 = keras.layers.MaxPool2D(pool_size=pool_size,
                                            strides=1,
                                            padding='same')
        self.pool3 = keras.layers.MaxPool2D(pool_size=pool_size,
                                            strides=1,
                                            padding='same')

    def call(self, inputs, training=None, mask=None):
        x0 = self.conv_in(inputs)
        x1 = self.pool1(x0)
        x2 = self.pool2(x1)
        x3 = self.pool3(x2)
        x = self.concat([x0, x1, x2, x3])
        x = self.conv_out(x)
        return x


class Bottleneck(keras.Model):
    def __init__(self, channels, shortcut, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)
        self.add = keras.layers.Add() if shortcut else None
        self.conv1 = Conv(channels, 3, 1, 1)
        self.conv2 = Conv(channels, 3, 1, 1)
        self.shortcut = shortcut

    def call(self, inputs, training=None, mask=None):
        if self.shortcut:
            x0 = self.conv1(inputs)
            x = self.conv2(x0)
            x = self.add([x0, x])
        else:
            x = self.conv1(inputs)
            x = self.conv2(x)
        return x


class C2F(keras.Model):
    def __init__(self, out_channels, num_bottlenecks, shortcut, *args, **kwargs):
        super(C2F, self).__init__(*args, **kwargs)
        self.conv_in = Conv(out_channels, 1, 1, 0)
        self.conv_out = Conv(out_channels, 1, 1, 0)
        self.concat = keras.layers.Concatenate()
        self.bottlenecks = [Bottleneck(int(0.5*out_channels), shortcut) for _ in range(num_bottlenecks)]

    def call(self, inputs, training=None, mask=None):
        x = self.conv_in(inputs)
        x, x0 = tf.split(x, 2, axis=-1)
        x_concat = [x, x0]
        for bottleneck in self.bottlenecks:
            x_concat.append(bottleneck(x_concat[-1]))
        x = self.concat(x_concat)
        x = self.conv_out(x)
        return x


class FeatureExtractor(keras.Model):
    def __init__(self, depth=1.0, width=1.0, ratio=1.0, *args, **kwargs):
        super(FeatureExtractor, self).__init__(*args, **kwargs)
        self.layer01_conv = Conv(int(64 * width), 3, 2, 1)
        self.layer02_conv = Conv(int(128 * width), 3, 2, 1)
        self.layer03_c2f = C2F(int(128 * width), int(3 * depth), True)
        self.layer04_conv = Conv(int(256 * width), 3, 2, 1)
        self.layer05_c2f = C2F(int(256 * width), int(6 * depth), True)
        self.layer06_conv = Conv(int(512 * width), 3, 2, 1)
        self.layer07_c2f = C2F(int(512 * width), int(6 * depth), True)
        self.layer08_conv = Conv(int(512 * width * ratio), 3, 2, 1)
        self.layer09_c2f = C2F(int(512 * width * ratio), int(3 * depth), True)
        self.layer10_sppf = SPPF(int(512 * width * ratio))

    def call(self, inputs, training=None, mask=None):
        x = self.layer01_conv(inputs)
        x = self.layer02_conv(x)
        x = self.layer03_c2f(x)
        x = self.layer04_conv(x)
        x = self.layer05_c2f(x)
        p3 = x
        x = self.layer06_conv(x)
        x = self.layer07_c2f(x)
        p4 = x
        x = self.layer08_conv(x)
        x = self.layer09_c2f(x)
        x = self.layer10_sppf(x)
        p5 = x
        return [p3, p4, p5]


class FPNUpSample(keras.Model):
    def __init__(self, out_channels, depth, *args, **kwargs):
        super(FPNUpSample, self).__init__(*args, **kwargs)
        self.up = keras.layers.UpSampling2D(2)
        self.concat = keras.layers.Concatenate()
        self.c2f = C2F(out_channels, int(3 * depth), False)

    def call(self, inputs, training=None, mask=None):
        start, target = inputs['start'], inputs['target']
        x = self.up(start)
        x = self.concat([x, target])
        x = self.c2f(x)
        return x


class FPNDownSample(keras.Model):
    def __init__(self, in_channels, out_channels, depth, *args, **kwargs):
        super(FPNDownSample, self).__init__(*args, **kwargs)
        self.conv = Conv(in_channels, 3, 2, 1)
        self.concat = keras.layers.Concatenate()
        self.c2f = C2F(out_channels, int(3 * depth), False)

    def call(self, inputs, training=None, mask=None):
        start, target = inputs['start'], inputs['target']
        x = self.conv(start)
        x = self.concat([x, target])
        x = self.c2f(x)
        return x


class FPN(keras.Model):
    def __init__(self, depth=1.0, width=1.0, ratio=1.0, *args, **kwargs):
        super(FPN, self).__init__(*args, **kwargs)
        self.up1 = FPNUpSample(int(512 * width), depth)
        self.up2 = FPNUpSample(int(256 * width), depth)
        self.down1 = FPNDownSample(int(256 * width), int(512 * width), depth)
        self.down2 = FPNDownSample(int(512 * width), int(512 * width * ratio), depth)

    def call(self, inputs, training=None, mask=None):
        p3, p4, p5 = inputs
        p4p5 = self.up1({'start': p5, 'target': p4})
        p3p4p5 = self.up2({'start': p4p5, 'target': p3})
        p3p4p5_d1 = self.down1({'start': p3p4p5, 'target': p4p5})
        p3p4p5_d2 = self.down2({'start': p3p4p5_d1, 'target': p5})
        return [p3p4p5, p3p4p5_d1, p3p4p5_d2]


class DetectionHead(keras.Model):
    def __init__(self, num_classes: int = 80, width=1.0, *args, **kwargs):
        # get_custom_objects().update({'ranged_leaky_relu': keras.layers.Activation(ranged_leaky_relu)})
        super(DetectionHead, self).__init__(*args, **kwargs)
        self.boxes1 = keras.Sequential(layers=[
            Conv(int(256 * width), 3, 1, 1),
            Conv(int(256 * width), 3, 1, 1),
            keras.layers.Conv2D(filters=64, kernel_size=1, strides=1)
        ])
        self.classes1 = keras.Sequential(layers=[
            Conv(int(256 * width), 3, 1, 1),
            Conv(int(256 * width), 3, 1, 1),
            keras.layers.Conv2D(filters=num_classes, kernel_size=1, strides=1),
            keras.layers.Activation(activation=tf.nn.sigmoid)
        ])
        self.distance_1 = keras.Sequential(layers=[
            Conv(int(256 * width), 3, 1, 1),
            Conv(int(256 * width), 3, 1, 1),
            keras.layers.Conv2D(filters=1, kernel_size=1, strides=1),
            keras.layers.Activation(activation='leaky_relu')
        ])
        self.concat1 = keras.layers.Concatenate(axis=-1)
        self.reshape1 = keras.layers.Reshape(target_shape=(-1, num_classes + 65))
        self.boxes2 = keras.Sequential(layers=[
            Conv(int(256 * width), 3, 1, 1),
            Conv(int(256 * width), 3, 1, 1),
            keras.layers.Conv2D(filters=64, kernel_size=1, strides=1)
        ])
        self.classes2 = keras.Sequential(layers=[
            Conv(int(256 * width), 3, 1, 1),
            Conv(int(256 * width), 3, 1, 1),
            keras.layers.Conv2D(filters=num_classes, kernel_size=1, strides=1),
            keras.layers.Activation(activation=tf.nn.sigmoid)
        ])
        self.distance_2 = keras.Sequential(layers=[
            Conv(int(256 * width), 3, 1, 1),
            Conv(int(256 * width), 3, 1, 1),
            keras.layers.Conv2D(filters=1, kernel_size=1, strides=1),
            keras.layers.Activation(activation='leaky_relu')
        ])
        self.concat2 = keras.layers.Concatenate(axis=-1)
        self.reshape2 = keras.layers.Reshape(target_shape=(-1, num_classes + 65))
        self.boxes3 = keras.Sequential(layers=[
            Conv(int(256 * width), 3, 1, 1),
            Conv(int(256 * width), 3, 1, 1),
            keras.layers.Conv2D(filters=64, kernel_size=1, strides=1)
        ])
        self.classes3 = keras.Sequential(layers=[
            Conv(int(256 * width), 3, 1, 1),
            Conv(int(256 * width), 3, 1, 1),
            keras.layers.Conv2D(filters=num_classes, kernel_size=1, strides=1),
            keras.layers.Activation(activation=tf.nn.sigmoid)
        ])
        self.distance_3 = keras.Sequential(layers=[
            Conv(int(256 * width), 3, 1, 1),
            Conv(int(256 * width), 3, 1, 1),
            keras.layers.Conv2D(filters=1, kernel_size=1, strides=1),
            keras.layers.Activation(activation='leaky_relu')
        ])
        self.concat3 = keras.layers.Concatenate(axis=-1)
        self.reshape3 = keras.layers.Reshape(target_shape=(-1, num_classes + 65))
        self.concat_out = keras.layers.Concatenate(axis=-2)
        self.act_out = keras.layers.Activation(activation='linear', dtype='float32')

    def call(self, inputs, training=None, mask=None):
        p3p4p5, p3p4p5_d1, p3p4p5_d2 = inputs
        x1_boxes = self.boxes1(p3p4p5)
        x1_classes = self.classes1(p3p4p5)
        x1_distance = self.distance_1(tf.concat([p3p4p5, x1_boxes, x1_classes], axis=-1))
        x1 = self.concat1([x1_boxes, x1_classes, x1_distance])
        x1 = self.reshape1(x1)
        x2_boxes = self.boxes2(p3p4p5_d1)
        x2_classes = self.classes2(p3p4p5_d1)
        x2_distance = self.distance_2(tf.concat([p3p4p5_d1, x2_boxes, x2_classes], axis=-1))
        x2 = self.concat2([x2_boxes, x2_classes, x2_distance])
        x2 = self.reshape2(x2)
        x3_boxes = self.boxes3(p3p4p5_d2)
        x3_classes = self.classes3(p3p4p5_d2)
        x3_distance = self.distance_3(tf.concat([p3p4p5_d2, x3_boxes, x3_classes], axis=-1))
        x3 = self.concat3([x3_boxes, x3_classes, x3_distance])
        x3 = self.reshape3(x3)
        x = self.concat_out([x1, x2, x3])
        x = self.act_out(x)
        return x


# TODO: implement SegmentationHead
class SegmentationHead(keras.Model):
    def __init__(self, *args, **kwargs):
        super(SegmentationHead, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        pass
