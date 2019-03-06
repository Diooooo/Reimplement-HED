import numpy as np
import tensorflow as tf

from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D
from keras.layers import Concatenate, Activation
from keras.models import Model
import keras.backend as K


def side_branch(x, factor):
    x = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(x)
    kernel_size = (2 * factor, 2 * factor)
    x = Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=factor, padding='same', use_bias=False)(x)
    return x


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x, dtype=dtype)
    # if x.dtype != dtype:
    #     x = tf.cast(x, dtype)  # change x.dtype
    return x


def cross_entropy_balanced(y_true, y_pred):  # tf tensor type
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=_epsilon, clip_value_max=1 - _epsilon)  # y_pred->[eps, 1-eps]
    y_pred = tf.log(y_pred / (1 - y_pred))  # from sigmoid(x) to x, where x is logit in tf

    y_true = tf.cast(y_true, tf.float32)

    neg = tf.reduce_sum(1 - y_true)  # edge pixels
    pos = tf.reduce_sum(y_true)  # non-edge pixels

    beta = neg / (neg + pos)

    pos_weight = beta / (1 - beta)

    # [target * -log(sigmoid(logit)) * pos_weight] + [(1-target)*log(1-sigmoid(logit))]
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pos_weight=pos_weight)
    loss = tf.reduce_mean(loss * (1 - beta))

    return tf.where(condition=tf.equal(loss, 0.0), x=0.0, y=loss)  # condition?x:y


def fuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='prediction')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def hed(load_model=False, file_path=None):
    img_input = Input(shape=(480, 480, 3), name='img_input')

    # stage 1
    # f()()--nested function
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv1')(img_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv2')(x)
    s1 = side_branch(x, 1)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool1')(x)

    # stage 2
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv1')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv2')(x)
    s2 = side_branch(x, 2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    # stage 3
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv1')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv2')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv3')(x)
    s3 = side_branch(x, 4)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

    # stage 4
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv3')(x)
    s4 = side_branch(x, 8)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)

    # stage 5
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv3')(x)
    s5 = side_branch(x, 16)

    # fuse layer
    fuse = Concatenate(axis=-1)([s1, s2, s3, s4, s5])  # axis=-1: last axis
    fuse = Conv2D(filters=1, kernel_size=(1, 1), padding='same', use_bias=False)(fuse)

    # outputs
    o_s1 = Activation(activation='sigmoid', name='o_s1')(s1)
    o_s2 = Activation(activation='sigmoid', name='o_s2')(s2)
    o_s3 = Activation(activation='sigmoid', name='o_s3')(s3)
    o_s4 = Activation(activation='sigmoid', name='o_s4')(s4)
    o_s5 = Activation(activation='sigmoid', name='o_s5')(s5)
    o_fuse = Activation(activation='sigmoid', name='o_fuse')(fuse)

    # define model graph
    model = Model(inputs=[img_input], outputs=[o_s1, o_s2, o_s3, o_s4, o_s5, o_fuse])
    # load predefined weights
    if load_model:
        model.load_weights(file_path, by_name=True)

    model.compile(loss={'o_s1': cross_entropy_balanced,
                        'o_s2': cross_entropy_balanced,
                        'o_s3': cross_entropy_balanced,
                        'o_s4': cross_entropy_balanced,
                        'o_s5': cross_entropy_balanced,
                        'o_fuse': cross_entropy_balanced,
                        },
                  metrics={'o_fuse': fuse_pixel_error},
                  optimizer='adam')
    return model


if __name__ == "__main__":
    model = hed(True, './model/vgg16_weights.h5')
