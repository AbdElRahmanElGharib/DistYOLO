import tensorflow as tf
import keras
# import tensorflow.keras as keras
import math


def maximum(x1, x2):
    if isinstance(x1, tf.SparseTensor):
        if isinstance(x2, tf.SparseTensor):
            return tf.sparse.maximum(x1, x2)
        else:
            x1 = tf.sparse.to_dense(x1)
    elif isinstance(x2, tf.SparseTensor):
        x2 = tf.sparse.to_dense(x2)
    return tf.math.maximum(x1, x2)


def minimum(x1, x2):
    if isinstance(x1, tf.SparseTensor):
        if isinstance(x2, tf.SparseTensor):
            return tf.sparse.minimum(x1, x2)
        else:
            x1 = tf.sparse.to_dense(x1)
    elif isinstance(x2, tf.SparseTensor):
        x2 = tf.sparse.to_dense(x2)
    return tf.math.minimum(x1, x2)
