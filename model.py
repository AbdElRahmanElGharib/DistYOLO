import tensorflow as tf
import keras
# import tensorflow.keras as keras


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
        x2 = self.pool1(x1)
        x3 = self.pool1(x2)
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
