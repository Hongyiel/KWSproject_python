import tensorflow as tf
import numpy as np
import sys


def main():
    input = tf.constant([[1, 2, 3],
                         [3, 4, 5],
                         [5, 6, 7],
                         [7, 8, 9]], dtype=tf.float32)
    filter = tf.constant([[1, 2],
                          [3, 4]], dtype=tf.float32)

    output = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
    tf.print(output, output_stream=sys.stderr)

if __name__ == "__main__":
    main()
