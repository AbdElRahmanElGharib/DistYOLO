import tensorflow as tf
import keras
# import tensorflow.keras as keras


class Conv(keras.Model):
    def __init__(self, filters, kernel_size, strides, padding,  *args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
        self.pad = None
        self.conv = None
        if isinstance(padding, str):
            if padding == 'same' or padding == 'valid':
                self.pad = keras.layers.ZeroPadding2D(padding=0)
                self.conv = keras.layers.Conv2D(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding=padding,
                                                use_bias=False)
            else:
                raise Exception(f'padding format not implemented. expected integer value\
                 or \'same\' or \'valid\'. found {padding}')
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
