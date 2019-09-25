import tensorflow as tf
import tf_encrypted as tfe
import numpy as np 

@tfe.local_computation(name_scope='provide_input')
def provide_input() -> tf.Tensor:
	# pick random tensor to be averaged
	return tf.random_normal(shape=(1,10))

@tfe.local_computation(name_scope='provide_input')
def provide_threshold() -> tf.Tensor:
	t = 0.5
	a=np.array([t] * 10)
	# print (a)
	b=tf.constant(a)
	return b


@tfe.local_computation('result-receiver', name_scope='receive_output')
def receive_output(average: tf.Tensor) -> tf.Operation:
	# simply print average
	return tf.print("Result:", average)

inputs = provide_input(player_name='inputter-0')
threshold = provide_threshold(player_name='threshold_inputter')
result = inputs - threshold

result_op = receive_output(result)

with tfe.Session() as sess:
	sess.run(result_op, tag='prune')
