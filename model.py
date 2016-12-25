#encoding:utf-8

import tensorflow as tf
from model_part import fc


def mlp(scope_name, x, input_dim, h1_dim, output_dim):
    with tf.variable_scope(scope_name) as scope:
        fc1 = fc('fc1', x, [input_dim, h1_dim], [h1_dim])
        # neural network output is approximate of q function
        q = fc('fc2', fc1, [h1_dim, output_dim], [output_dim])
    return q