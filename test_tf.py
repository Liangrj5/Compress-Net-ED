import sys
import logging

import tensorflow as tf
import tf_encrypted as tfe
from convert import decode
import tensorflow.keras as keras

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



class ModelOwner():
	"""Contains code meant to be executed by the model owner.

	Args:
		player_name: `str`, name of the `tfe.player.Player`
								 representing the model owner.
		local_data_file: filepath to MNIST data.
	"""
	BATCH_SIZE = 128
	NUM_CLASSES = 10
	EPOCHS = 1

	ITERATIONS = 60000 // BATCH_SIZE

	IMG_ROWS = 28
	IMG_COLS = 28
	FLATTENED_DIM = IMG_ROWS * IMG_COLS

	def __init__(self, player_name, local_data_file):
		self.player_name = player_name
		self.local_data_file = local_data_file

	def _build_data_pipeline(self): #-> plaintext data iterator
		"""Build a reproducible tf.data iterator."""

		def normalize(image, label):
			image = tf.cast(image, tf.float32) / 255.0
			return image, label

		def flatten(image, label):
			image = tf.reshape(image, shape=[self.FLATTENED_DIM])
			return image, label

		dataset = tf.data.TFRecordDataset([self.local_data_file])
		dataset = dataset.map(decode)
		dataset = dataset.map(normalize)
		dataset = dataset.map(flatten)
		dataset = dataset.repeat()
		dataset = dataset.batch(self.BATCH_SIZE)

		iterator = dataset.make_one_shot_iterator()
		return iterator 

	def _build_training_graph(self, training_data):
		"""Build a graph for plaintext model training."""

		model = keras.Sequential()
		model.add(keras.layers.Dense(512, input_shape=[self.FLATTENED_DIM,]))
		model.add(keras.layers.Activation('relu'))
		model.add(keras.layers.Dense(self.NUM_CLASSES, activation=None))

		# optimizer and data pipeline
		optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

		def loss(model, inputs, targets):
			logits = model(inputs)
			per_element_loss = tf.losses.sparse_softmax_cross_entropy(
					labels=targets, logits=logits)
			return tf.reduce_mean(per_element_loss)

		def grad(model, inputs, targets):
			loss_value = loss(model, inputs, targets)
			return loss_value, tf.gradients(loss_value, model.trainable_variables)

		def loop_body(i):
			print(i)
			x, y = training_data.get_next()
			_, grads = grad(model, x, y)
			update_op = optimizer.apply_gradients(
					zip(grads, model.trainable_variables))
			with tf.control_dependencies([update_op]):
				return i + 1

		loop = tf.while_loop(lambda i: i < self.ITERATIONS * self.EPOCHS,
												 loop_body, loop_vars=(0,))

		with tf.control_dependencies([loop]):
			print_op = tf.print("Training complete")
		with tf.control_dependencies([print_op]):
			return [tf.identity(x) for x in model.trainable_variables]

	# @tfe.local_computation
	def provide_weights(self):
		print('*' * 40)
		with tf.name_scope('loading'):
			training_data = self._build_data_pipeline()

		with tf.name_scope('training'):
			parameters = self._build_training_graph(training_data)

		return parameters


if __name__ == "__main__":


	model_owner = ModelOwner(
			player_name="model-owner",
			local_data_file="./data/train.tfrecord")

	# get model parameters as private tensors from model owner

	params = model_owner.provide_weights()
	# with tf.Session() as sess:
	# 	# sess.run(tfe.global_variables_initializer())
	# 	print(sess.run(params[3]))
	print(params)
	print(params[3])
	c  = params[3]


	# with tfe.Session() as sess:
	#     # initialize variables
	#     # reveal result
	#     result = sess.run(y.reveal())