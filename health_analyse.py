import numpy as np
import tensorflow as tf

snack_tensor = tf.constant([7, 8, 3, 5], dtype=tf.float32)
weights_tensor = tf.constant([0.5, 0.7, 0.3, 0.2], dtype=tf.float32)

weighted_inputs = tf.multiply(snack_tensor, weights_tensor)
total_input = tf.reduce_sum(weighted_inputs)
result = tf.nn.relu(total_input)
end_result = tf.round(result)

tf.print('result:', end_result)