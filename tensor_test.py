import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
np.set_printoptions(precision=3)
np.random.seed(102)


def main():
    conv_filter = tf.ones((10, 4, 1, 64))
    depth_filter = tf.ones((3, 3, 64, 1))
    point_filter = tf.ones((1, 1, 64, 64))
    input = tf.ones((1, 49, 10, 1))
    input1 = tf.keras.layers.ZeroPadding2D(padding=((4, 5), (1, 1)))(input)
    output1 = tf.nn.conv2d(input1, conv_filter, strides=[
                          1, 2, 2, 1], padding='VALID', data_format='NHWC')
    input2 = tf.nn.relu(output1)
    # print(input2.shape)

    # output = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(output)
    output2 = tf.nn.separable_conv2d(input2, depth_filter, point_filter, strides=[
                                    1, 1, 1, 1], padding='SAME', data_format='NHWC')
    input3 = tf.nn.relu(output2)
    # print(input3)


if __name__ == "__main__":
    main()
