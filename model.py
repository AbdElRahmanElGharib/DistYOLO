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
