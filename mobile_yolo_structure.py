from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Add, \
    GlobalAveragePooling2D, Reshape, Multiply, MaxPool2D, Concatenate, UpSampling2D, Input, Conv1D


BOX_REGRESSORS = 64


def gcd(a, b):
    if a == 0:
        return b
    return gcd(b % a, a)


class SqueezeExcite(Model):
    def __init__(self, input_channels, reduction_ratio=4, grouping_factor=1):
        super(SqueezeExcite, self).__init__()
        self.global_avg_pool = GlobalAveragePooling2D()
        self.reshape = Reshape((1, 1, input_channels))
        self.fc1 = Conv2D(input_channels // reduction_ratio, kernel_size=1, activation='relu', use_bias=False,
                          groups=int(gcd(input_channels // reduction_ratio, input_channels) / grouping_factor))
        self.fc2 = Conv2D(input_channels, kernel_size=1, activation='sigmoid', use_bias=False,
                          groups=int(gcd(input_channels // reduction_ratio, input_channels) / grouping_factor))

    def call(self, inputs, training=None, mask=None):
        x = self.global_avg_pool(inputs)
        x = self.reshape(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return Multiply()([inputs, x])


class BottleneckBlock(Model):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, use_se, nl, grouping_factor=1):
        super(BottleneckBlock, self).__init__()
        self.use_residual = (in_channels == out_channels and stride == 1)
        mid_channels = in_channels * expansion_factor

        self.expand = Conv2D(mid_channels, kernel_size=1, use_bias=False,
                             groups=int(gcd(mid_channels, in_channels) // grouping_factor))
        self.bn1 = BatchNormalization()
        self.depth_wise_conv = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        if use_se:
            self.se = SqueezeExcite(mid_channels, grouping_factor=grouping_factor)
        else:
            self.se = None
        self.compress = Conv2D(out_channels, kernel_size=1, use_bias=False,
                               groups=int(gcd(mid_channels, out_channels) // grouping_factor))
        self.bn3 = BatchNormalization()
        self.activation = Activation('relu' if nl == 'RE' else 'swish')

    def call(self, inputs, training=None, mask=None):
        x = self.expand(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)

        x = self.depth_wise_conv(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)

        if self.se is not None:
            x = self.se(x)

        x = self.compress(x)
        x = self.bn3(x, training=training)

        if self.use_residual:
            x = Add()([x, inputs])

        return x


class MobileNetV3Backbone(Model):
    def __init__(self, grouping_factor=1):
        super(MobileNetV3Backbone, self).__init__()

        self.conv1 = Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.act1 = Activation('swish')

        self.bottleneck1 = BottleneckBlock(16, 16, 1, 2, use_se=True, nl='RE', grouping_factor=grouping_factor)
        self.bottleneck2 = BottleneckBlock(16, 24, 4.5, 2, use_se=False, nl='RE', grouping_factor=grouping_factor)
        self.bottleneck3 = BottleneckBlock(24, 24, 3.66, 1, use_se=False, nl='RE', grouping_factor=grouping_factor)
        self.bottleneck4 = BottleneckBlock(24, 40, 4, 2, use_se=True, nl='HS', grouping_factor=grouping_factor)
        self.bottleneck5 = BottleneckBlock(40, 40, 6, 1, use_se=True, nl='HS', grouping_factor=grouping_factor)
        self.bottleneck6 = BottleneckBlock(40, 48, 3, 1, use_se=True, nl='HS', grouping_factor=grouping_factor)
        self.bottleneck7 = BottleneckBlock(48, 48, 3, 1, use_se=True, nl='HS', grouping_factor=grouping_factor)
        self.bottleneck8 = BottleneckBlock(48, 96, 6, 2, use_se=True, nl='HS', grouping_factor=grouping_factor)

        self.conv2 = Conv2D(576, kernel_size=1, use_bias=False, groups=int(96 // grouping_factor))
        self.bn2 = BatchNormalization()
        self.act2 = Activation('swish')

        self.build((None, 320, 320, 3))

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.bottleneck1(x, training=training)
        x = self.bottleneck2(x, training=training)
        x1 = self.bottleneck3(x, training=training)
        x = self.bottleneck4(x1, training=training)
        x = self.bottleneck5(x, training=training)
        x2 = self.bottleneck6(x, training=training)
        x = self.bottleneck7(x2, training=training)
        x = self.bottleneck8(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x3 = self.act2(x)

        # shapes: x1 -> (40, 40, 24), x2 -> (20, 20, 48), x3 -> (10, 10, 576)
        return x1, x2, x3


class DetectionHead(Model):
    def __init__(self, num_classes=80, grouping_factor=1, *args, **kwargs):
        super(DetectionHead, self).__init__(*args, **kwargs)

        self.sppf_1 = MaxPool2D(5, 1, padding='same')
        self.sppf_2 = MaxPool2D(9, 1, padding='same')
        self.sppf_3 = MaxPool2D(13, 1, padding='same')
        self.sppf_4 = Conv2D(576, 1, groups=64)
        self.sppf_concat = Concatenate()

        self.conv_in_route1 = Conv2D(256, 1, use_bias=False, groups=int(64 // grouping_factor))
        self.bn_in_route1 = BatchNormalization()
        self.act_in_route1 = Activation('leaky_relu')

        self.bottleneck1_route1 = BottleneckBlock(256, 256, 1, 1, False, 'RE', grouping_factor=grouping_factor)
        self.bottleneck2_route1 = BottleneckBlock(256, 256, 1, 1, False, 'RE', grouping_factor=grouping_factor)

        self.conv_out_route1 = Conv2D((num_classes + BOX_REGRESSORS), 1, use_bias=False,
                                      groups=int(gcd(256, (num_classes + BOX_REGRESSORS)) // grouping_factor))
        self.bn_out_route1 = BatchNormalization()
        self.act_out_route1 = Activation('leaky_relu')
        self.reshape_route1 = Reshape((-1, (num_classes + BOX_REGRESSORS)))

        self.conv_up_1 = Conv2D(128, 1, use_bias=False, groups=int(128 // grouping_factor))
        self.bn_up_1 = BatchNormalization()
        self.act_up_1 = Activation('leaky_relu')
        self.up_1 = UpSampling2D(size=2)

        self.concat_route2 = Concatenate()

        self.conv_in_route2 = Conv2D(128, 1, use_bias=False, groups=16)
        self.bn_in_route2 = BatchNormalization()
        self.act_in_route2 = Activation('leaky_relu')

        self.bottleneck1_route2 = BottleneckBlock(128, 128, 1, 1, False, 'RE', grouping_factor=grouping_factor)
        self.bottleneck2_route2 = BottleneckBlock(128, 128, 1, 1, False, 'RE', grouping_factor=grouping_factor)

        self.conv_out_route2 = Conv2D((num_classes + BOX_REGRESSORS), 1, use_bias=False,
                                      groups=int(gcd(128, (num_classes + BOX_REGRESSORS)) // grouping_factor))
        self.bn_out_route2 = BatchNormalization()
        self.act_out_route2 = Activation('leaky_relu')
        self.reshape_route2 = Reshape((-1, (num_classes + BOX_REGRESSORS)))

        self.conv_up_2 = Conv2D(64, 1, use_bias=False, groups=int(64 // grouping_factor))
        self.bn_up_2 = BatchNormalization()
        self.act_up_2 = Activation('leaky_relu')
        self.up_2 = UpSampling2D(size=2)

        self.concat_route3 = Concatenate()

        self.conv_in_route3 = Conv2D(64, 1, use_bias=False, groups=8)
        self.bn_in_route3 = BatchNormalization()
        self.act_in_route3 = Activation('leaky_relu')

        self.bottleneck1_route3 = BottleneckBlock(64, 64, 1, 1, False, 'RE', grouping_factor=grouping_factor)
        self.bottleneck2_route3 = BottleneckBlock(64, 64, 1, 1, False, 'RE', grouping_factor=grouping_factor)

        self.conv_out_route3 = Conv2D((num_classes + BOX_REGRESSORS), 1, use_bias=False,
                                      groups=int(gcd(64, (num_classes + BOX_REGRESSORS)) // grouping_factor))
        self.bn_out_route3 = BatchNormalization()
        self.act_out_route3 = Activation('leaky_relu')
        self.reshape_route3 = Reshape((-1, (num_classes + BOX_REGRESSORS)))

        self.concat_out = Concatenate(axis=-2)

        self.build([(None, 40, 40, 24), (None, 20, 20, 48),  (None, 10, 10, 576)])

    def call(self, inputs, training=None, mask=None):
        x1, x2, x3 = inputs

        sppf1 = self.sppf_1(x3)
        sppf2 = self.sppf_2(x3)
        sppf3 = self.sppf_3(x3)
        sppf4 = self.sppf_4(x3)

        sppf = self.sppf_concat([sppf1, sppf2, sppf3, sppf4])

        x = self.conv_in_route1(sppf)
        x = self.bn_in_route1(x, training=training)
        x = self.act_in_route1(x)

        x = self.bottleneck1_route1(x, training=training)
        x_mid = self.bottleneck2_route1(x, training=training)

        x = self.conv_out_route1(x_mid)
        x = self.bn_out_route1(x, training=training)
        x = self.act_out_route1(x)

        out_1 = self.reshape_route1(x)

        x = self.conv_up_1(x_mid)
        x = self.bn_up_1(x, training=training)
        x = self.act_up_1(x)
        x = self.up_1(x)

        x = self.concat_route2([x, x2])

        x = self.conv_in_route2(x)
        x = self.bn_in_route2(x, training=training)
        x = self.act_in_route2(x)

        x = self.bottleneck1_route2(x, training=training)
        x_mid = self.bottleneck2_route2(x, training=training)

        x = self.conv_out_route2(x_mid)
        x = self.bn_out_route2(x, training=training)
        x = self.act_out_route2(x)

        out_2 = self.reshape_route2(x)

        x = self.conv_up_2(x_mid)
        x = self.bn_up_2(x, training=training)
        x = self.act_up_2(x)
        x = self.up_2(x)

        x = self.concat_route3([x, x1])

        x = self.conv_in_route3(x)
        x = self.bn_in_route3(x, training=training)
        x = self.act_in_route3(x)

        x = self.bottleneck1_route3(x, training=training)
        x = self.bottleneck2_route3(x, training=training)

        x = self.conv_out_route3(x)
        x = self.bn_out_route3(x, training=training)
        x = self.act_out_route3(x)

        out_3 = self.reshape_route3(x)

        out = self.concat_out([out_1, out_2, out_3])

        return out


class DistanceHead(Model):
    def __init__(self, num_classes=80, grouping_factor=1, *args, **kwargs):
        super(DistanceHead, self).__init__(*args, **kwargs)

        self.pre_conv1 = Conv2D(256, 1, groups=64)
        self.pre_conv2 = Conv2D(256, 1, groups=16)
        self.pre_conv3 = Conv2D(256, 1, groups=8)

        self.pre_reshape1 = Reshape((-1, 256))
        self.pre_reshape2 = Reshape((-1, 256))
        self.pre_reshape3 = Reshape((-1, 256))

        self.pre_concat = Concatenate(axis=-2)

        self.concat = Concatenate()

        self.conv1 = Conv1D(256, 1, use_bias=False,
                            groups=int(gcd(256, 256 + num_classes + BOX_REGRESSORS) // grouping_factor))
        self.bn1 = BatchNormalization()
        self.act1 = Activation('leaky_relu')

        self.conv2 = Conv1D(128, 1, use_bias=False, groups=int(128 // grouping_factor))
        self.bn2 = BatchNormalization()
        self.act2 = Activation('leaky_relu')

        self.conv3 = Conv1D(128, 1, use_bias=False, groups=int(128 // grouping_factor))
        self.bn3 = BatchNormalization()
        self.act3 = Activation('leaky_relu')

        self.conv_out = Conv1D(1, 1)
        self.act_out = Activation('linear')

        self.build([(None, 40, 40, 24), (None, 20, 20, 48), (None, 10, 10, 576),
                    (None, 2100, (num_classes+BOX_REGRESSORS))])

    def call(self, inputs, training=None, mask=None):
        x1, x2, x3, x_detections = inputs

        x3_pre = self.pre_conv1(x3)
        x3_pre = self.pre_reshape1(x3_pre)

        x2_pre = self.pre_conv2(x2)
        x2_pre = self.pre_reshape2(x2_pre)

        x1_pre = self.pre_conv3(x1)
        x1_pre = self.pre_reshape3(x1_pre)

        x_pre = self.pre_concat([x3_pre, x2_pre, x1_pre])

        x = self.concat([x_detections, x_pre])

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)

        x = self.conv_out(x)
        x = self.act_out(x)

        return x


class Conv(Model):
    def __init__(self, filters, kernel_size, groups=1, dilation_rate=1, *args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
        self.depth_wise = DepthwiseConv2D(kernel_size=kernel_size, groups=groups,
                                          padding='same', dilation_rate=dilation_rate, use_bias=False)
        self.point_wise = Conv2D(filters=filters, kernel_size=1, padding='same', groups=groups, use_bias=False)
        self.bn = BatchNormalization()
        self.act = Activation('leaky_relu')

    def call(self, inputs, training=None, mask=None):
        x = self.depth_wise(inputs)
        x = self.point_wise(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x


class SegmentationHead(Model):
    def __init__(self, num_classes=1, grouping_factor=1, *args, **kwargs):
        super(SegmentationHead, self).__init__(*args, **kwargs)

        self.conv1 = Conv(filters=256, kernel_size=3, groups=int(64 // grouping_factor), dilation_rate=2)

        self.up1 = UpSampling2D(size=4)
        self.up1_in = UpSampling2D(size=2)
        self.up1_conv_in = DepthwiseConv2D(kernel_size=1, depth_multiplier=4, groups=48, use_bias=False)
        self.up1_concat = Concatenate()

        self.conv2 = Conv(filters=256, kernel_size=5, groups=64, dilation_rate=2)

        self.up2_conv_in = DepthwiseConv2D(kernel_size=1, depth_multiplier=8, groups=24, use_bias=False)
        self.up2_concat = Concatenate()
        self.up2 = UpSampling2D(size=4)

        self.conv3 = Conv(filters=256, kernel_size=5, groups=64, dilation_rate=2)
        self.up3 = UpSampling2D(size=2)

        self.conv4 = Conv(filters=256, kernel_size=5, groups=int(256 // grouping_factor), dilation_rate=2)
        self.image_concat = Concatenate()
        self.conv5 = Conv(filters=148, kernel_size=5, groups=37, dilation_rate=2)

        self.conv_out = Conv2D(filters=num_classes, kernel_size=1)
        self.act_out = Activation('sigmoid')

        self.build([(None, 320, 320, 3), (None, 40, 40, 24), (None, 20, 20, 48), (None, 10, 10, 576)])

    def call(self, inputs, training=None, mask=None):
        image_input, x1, x2, x3 = inputs

        x = self.conv1(x3, training=training)

        x = self.up1(x)
        x2 = self.up1_in(x2)
        x2 = self.up1_conv_in(x2)
        x = self.up1_concat([x, x2])

        x = self.conv2(x, training=training)

        x1 = self.up2_conv_in(x1)
        x = self.up2_concat([x, x1])
        x = self.up2(x)

        x = self.conv3(x, training=training)
        x = self.up3(x)

        x = self.conv4(x, training=training)
        x = self.image_concat([x, image_input])
        x = self.conv5(x, training=training)

        x = self.conv_out(x)
        x = self.act_out(x)

        return x


# Example usage
if __name__ == '__main__':
    input_image = Input(shape=(320, 320, 3), name='input_image')
    features = MobileNetV3Backbone()(input_image)
    detections = DetectionHead()(features)
    distances = DistanceHead()((*features, detections))
    segments = SegmentationHead()((input_image, *features))
    model = Model(inputs=[input_image], outputs=[detections, distances, segments],
                  name='MobileYOLO_with_distance_and_segmentation')
    model.summary(expand_nested=True)
    # 145k parameters
