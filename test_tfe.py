import tensorflow as tf
import tf_encrypted as tfe
import logging

session_target = None


class ModelOwner():
	def __init__(self, player_name, local_data_file):
		self.player_name = player_name
		self.local_data_file = local_data_file


  @tfe.local_computation
  def provide_weights(self):
    with tf.name_scope('loading'):
      training_data = self._build_data_pipeline()

    with tf.name_scope('training'):
      parameters = self._build_training_graph(training_data)

    return parameters


class PredictionClient():
	def __init__(self, player_name, local_data_file):
		self.player_name = player_name
		self.local_data_file = local_data_file
	def _build_training_graph(self, training_data):
	"""Build a graph for plaintext model training."""

	model = keras.Sequential()
	model.add(keras.layers.Dense(512, input_shape=[self.FLATTENED_DIM,]))
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dense(self.NUM_CLASSES, activation=None))
	res = []
	for x in model.trainable_variables:
		res.append(tf.random_normal(shape=x.size()))


logging.basicConfig(level=logging.DEBUG)


model_owner = ModelOwner(
		player_name="model-owner",
		local_data_file="./data/train.tfrecord")

prediction_client = PredictionClient(
		player_name="prediction-client",
		local_data_file="./data/test.tfrecord")


 params = model_owner.provide_weights()
