import tensorflow as tf
import tf_encrypted as tfe
import logging

session_target = None
import tensorflow.keras as keras
import tensorflow as tf
from convert import decode
import tensorflow.keras as keras
import numpy as np
print("**" * 15 , 'STAT', "**"*15)
print("**" * 15 , 'STAT', "**"*15)
print("**" * 15 , 'STAT', "**"*15)
class ModelOwner():

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

	def _build_training_graph(self, training_data):


		model = keras.Sequential()
		model.add(keras.layers.Dense(512, input_shape=[self.FLATTENED_DIM,]))
		model.add(keras.layers.Activation('relu'))
		model.add(keras.layers.Dense(self.NUM_CLASSES, activation=None))
		res = []
		for x in model.trainable_variables:
			temp = np.random.rand( * x.get_shape().as_list())
			res.append(tf.convert_to_tensor(temp))
		return res

	# @tfe.local_computation
	def provide_weights(self):
		training_data = None
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
	c	= params[3]

