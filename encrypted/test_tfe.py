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

		b = np.load("paramters.npy", allow_pickle=True)

		res = []
		for index, x in enumerate(model.trainable_variables):
			print(x)
			# input()
			#temp = np.random.rand( * x.get_shape().as_list())
			temp = b[index]
			res.append(tf.convert_to_tensor(temp))

		# b = np.load("paramters.npy", allow_pickle=True)
		# res  = []
		# for i in b:
		# 	input()
		#     # res.append(i)
		# 	res.append(tf.convert_to_tensor(i))

		return res

	@tfe.local_computation
	def provide_weights(self):
		training_data = None
		with tf.name_scope('training'):
			parameters = self._build_training_graph(training_data)

		return parameters


class PredictionClient():
	"""
	Contains code meant to be executed by a prediction client.

	Args:
		player_name: `str`, name of the `tfe.player.Player`
								 representing the data owner
		build_update_step: `Callable`, the function used to construct
											 a local federated learning update.
	"""

	BATCH_SIZE = 20

	def __init__(self, player_name, local_data_file):
		self.player_name = player_name
		self.local_data_file = local_data_file

	def _build_data_pipeline(self):
		"""Build a reproducible tf.data iterator."""

		def normalize(image, label):
			image = tf.cast(image, tf.float32) / 255.0
			return image, label

		dataset = tf.data.TFRecordDataset([self.local_data_file])
		dataset = dataset.map(decode)
		dataset = dataset.map(normalize)
		dataset = dataset.repeat()
		dataset = dataset.batch(self.BATCH_SIZE)

		iterator = dataset.make_one_shot_iterator()
		return iterator

	@tfe.local_computation
	def provide_input(self) -> tf.Tensor:
		"""Prepare input data for prediction."""
		with tf.name_scope('loading'):
			prediction_input, expected_result = self._build_data_pipeline().get_next()
			print_op = tf.print("Expect", expected_result, summarize=self.BATCH_SIZE)
			with tf.control_dependencies([print_op]):
				prediction_input = tf.identity(prediction_input)

		with tf.name_scope('pre-processing'):
			prediction_input = tf.reshape(
					prediction_input, shape=(self.BATCH_SIZE, ModelOwner.FLATTENED_DIM))
		return prediction_input

	@tfe.local_computation
	def receive_output(self, logits: tf.Tensor) -> tf.Operation:
		with tf.name_scope('post-processing'):
			prediction = tf.argmax(logits, axis=1)
			op = tf.print("Result", prediction, summarize=self.BATCH_SIZE)
			return op


if __name__ == "__main__":

	logging.basicConfig(level=logging.DEBUG)

	model_owner = ModelOwner(
			player_name="model-owner",
			local_data_file="./data/train.tfrecord")

	prediction_client = PredictionClient(
			player_name="prediction-client",
			local_data_file="./data/test.tfrecord")

	# get model parameters as private tensors from model owner
	params = model_owner.provide_weights()

	# we'll use the same parameters for each prediction so we cache them to
	# avoid re-training each time
	cache_updater, params = tfe.cache(params)

	with tfe.protocol.SecureNN():
		# get prediction input from client
		x = prediction_client.provide_input()

		model = tfe.keras.Sequential()
		model.add(tfe.keras.layers.Dense(512, batch_input_shape=x.shape))
		model.add(tfe.keras.layers.Activation('relu'))
		model.add(tfe.keras.layers.Dense(10, activation=None))

		logits = model(x)

	# send prediction output back to client
	prediction_op = prediction_client.receive_output(logits)

	with tfe.Session(target=session_target) as sess:
		sess.run(tf.global_variables_initializer(), tag='init')

		print("Training")
		sess.run(cache_updater, tag='training')

		print("Set trained weights")
		model.set_weights(params, sess)

		for _ in range(5):
			print("Predicting")
			sess.run(prediction_op, tag='prediction')
