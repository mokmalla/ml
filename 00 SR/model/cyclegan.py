from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Dense, Flatten, LeakyReLU, ReLU, Concatenate, Layer
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf

from model.common import normalize, denormalize, pixel_shuffle

class PixelShuffle(Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config.update({"scale": self.scale})
        return config

def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        x = PixelShuffle(scale=factor)(x)
        return x

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x

def GAN_res_block(x_in, num_filters, momentum=0.8, skip_connection=True):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    if skip_connection == True:
        x = Add()([x_in, x])
    return x


def HR_generator(scale=4, num_filters=64, num_res_blocks=3):
    x_in = Input(shape=(None, None, 3))
    x = x2 = Conv2D(num_filters, 3, padding='same')(x_in)

    d_x = []
    for _ in range(num_res_blocks):
        x = GAN_res_block(x, num_filters)
        d_x.append(x)

    x = GAN_res_block(x, num_filters, skip_connection=False)

    d_x.append(x)
    x = Concatenate()(d_x)

    x = Conv2D(num_filters, kernel_size=1, padding='same')(x)
    x = Add()([x2, x])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    return Model(x_in, x, name="gan_HR")

def discriminator_block(x_in, num_filters, strides=1):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    return LeakyReLU(alpha=0.2)(x)

def discriminator(num_filters=64):
    x_in = Input(shape=(None, None, 3))

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)

    x = discriminator_block(x, num_filters)
    x = discriminator_block(x, num_filters)
    x = discriminator_block(x, num_filters)
    x = discriminator_block(x, num_filters)
    x = discriminator_block(x, num_filters, 2)

    x = discriminator_block(x, num_filters)

    return Model(x_in, x)

def LR_generator(num_filters=64):
    x_in = Input(shape=(None, None, 3))

    x = Conv2D(num_filters, 3, strides = 2, padding='same', activation='relu')(x_in) # downscaling
    x = Conv2D(num_filters, 3, strides = 2, padding='same', activation='relu')(x) # downscaling
    skip1 = x

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    skip2 = x

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)

    x = Concatenate()([x, skip2])
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)

    x = Add()([skip1, x])

    x = Conv2D(3, 3, padding='same', activation='relu')(x)

    return Model(x_in, x)