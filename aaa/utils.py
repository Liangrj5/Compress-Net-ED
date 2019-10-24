import tensorflow as tf
import tf_encrypted as tfe
import numpy as np
from scipy import sparse
import os

# from tf_encrypted.protocol.securenn import SecureNN
# from tf_encrypted.protocol.protocol import get_protocol, set_protocol
# from tf_encrypted.protocol.pond.pond import Pond, PondPrivateTensor


def get_private_avg(x_private):
	nums = int(np.prod(x_private.shape))
	temp = x_private
	while sum(temp.shape) > 1:
		temp = temp.reduce_sum(axis=-1)
	res = temp / nums

	# with tfe.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	print('x_private	\n', sess.run(x_private.reveal()))
	# 	print('temp	\n', sess.run(temp.reveal()))
	# 	print('res	\n', sess.run(res.reveal()))
	return res
	# add_n
def get_private_var(x_private):
	nums = int(np.prod(x_private.shape))
	x_avg = get_private_avg(x_private)
	x = x_private - x_avg
	x = x * x
	while sum(x.shape) > 1:
		x = x.reduce_sum(axis=-1)

	x_var = x / nums

	# with tfe.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	print('x_private	\n', sess.run(x_private.reveal()))
	# 	print('x_avg	\n', sess.run(x_avg.reveal()))
	# 	print('x_var	\n', sess.run(x_var.reveal()))
	return x_var

def get_weight_mask(x_private, threshold, prot):
	'''
	x_private: PondPrivateVariable
	threshold: int
	prot: SecureNN

	return: PondPublicVariable
	'''
	x_var = get_private_var(x_private)
	t = (threshold **2 ) * x_var 

	x_2 = x_private.square()
	# x_2 = x_2 - t
	weight_mask = prot.greater(x_2, t)
	weight_mask = weight_mask.reveal()

	# with tfe.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	print('weight_mask	\n', sess.run(weight_mask))
	# with tfe.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	print('x_private	\n', sess.run(x_private.reveal()))
	# 	np_var = sess.run(t.reveal())
	# 	print('threshold	\n', np_var ** 0.5 )
	# 	print('weight_mask	\n', sess.run(weight_mask.reveal()))
	return weight_mask

def _private_sqrt():
	pass

# get_private_avg(w_private)
# get_private_var(w_private)
# res = get_private_sparse_matrix(w_private, mask_public)
# save_private_sparse_matrix(res)
# load_private_sparse_matrix(1,res.shape)

# with tfe.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	print('w_private	\n', sess.run(w_private.reveal()))
# 	print('mask_reveal\n', sess.run(mask_public))
# 	print('w_mask		 \n', sess.run(w_mask.reveal()))
# 	print('w_mask0		\n', sess.run(w_mask.reveal().value_on_0.value))
# 	print('w_mask1		\n', sess.run(w_mask.reveal().value_on_1.value))
# 	print('w_private0		\n', sess.run(w_mask.share0.value))
# 	print('w_private1		\n', sess.run(w_mask.share0.value))

def get_private_sparse_matrix(w_private, mask_public):
	w_private0, w_private1 = w_mask.unwrapped
	mask_public0, mask_public1 = mask_public.unwrapped

	w_mask0 = mask_public0 * w_private0
	w_mask1 = mask_public1 * w_private1

	# w_mask0 = prot.tensor_factory.tensor(w_mask0)
	# w_mask1 = prot.tensor_factory.tensor(w_mask1)
	res = PondPrivateTensor(prot =w_private.prot,
		share0 = w_mask0,
		share1 = w_mask1,
		is_scaled = False)
	res.is_scaled = w_private.is_scaled
	# with tfe.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
		# print('w_mask0',w_mask0)
		# print('w_mask1',w_mask1.value)
		# print('w_private		\n', sess.run(w_private.reveal()))
		# print('w_mask0		\n', sess.run(ccc.reveal()))
		# print('w_mask1		\n', sess.run(ccc.reveal()))
		# c1, c2 = ccc.unwrapped
		# print('c1		\n', sess.run(c1))
		# print('c2		\n', sess.run(c2))

	return res



def save_private_sparse_matrix(w_mask_private,  save_name):
	w_private0, w_private1 = w_mask_private.unwrapped

	with tfe.Session() as sess:
		sess.run(tf.global_variables_initializer())
		temp0 = sess.run(w_private0.value)
		temp1 = sess.run(w_private1.value)

	save_name0 = save_name + '0'
	save_name1 = save_name + '1'

	temp0 = np.reshape(temp0,[temp0.shape[0],-1])
	temp0 = sparse.csc_matrix(temp0)
	sparse.save_npz(save_name0, temp0)		
	temp1 = np.reshape(temp1,[temp1.shape[0],-1])
	temp1 = sparse.csc_matrix(temp1)
	sparse.save_npz(save_name1, temp1)

	# with tfe.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	print('w_mask_private		\n', sess.run(w_mask_private.reveal()))

def save_private_dense_matrix(w_private, save_name):
	w_private0, w_private1 = w_private.unwrapped

	save_name0 = save_name + '0'
	save_name1 = save_name + '1'
	with tfe.Session() as sess:
		sess.run(tf.global_variables_initializer())
		temp0 = sess.run(w_private0.value)
		temp1 = sess.run(w_private1.value)

	np.save(save_name0,temp0)
	np.save(save_name1,temp1)


def save_model_sparse_weight(model, save_dir):

	for layer in (model.layers):
		name = os.path.join(save_dir, layer.name)
		# print(name)
		if 'dense' in layer.name:
			weight = layer.weights[0]
			bais = layer.weights[1]
			save_private_sparse_matrix(weight, name + '_weight')
			save_private_dense_matrix(bais, name + '_bais')
		else:
			for i , weight in enumerate(layer.weights):
				save_private_dense_matrix(weight, name + '_w' + str(i) + '_')


def save_model_dense_weight(model, save_dir):

	for layer in (model.layers):
		name = os.path.join(save_dir, layer.name)
		for i , weight in enumerate(layer.weights):
			save_private_dense_matrix(weight, name + '_w' + str(i) + '_')



def load_private_sparse_matrix(files_name, shape):
	temp0 = sparse.load_npz('./tmp/sp_matrix0.npz')
	temp1 = sparse.load_npz('./tmp/sp_matrix1.npz')
	temp0 = temp0.todense().A
	temp1 = temp1.todense().A
	temp0 = np.reshape(temp0, shape)
	temp1 = np.reshape(temp1, shape)

	temp0 = prot.tensor_factory.tensor(temp0)
	temp1 = prot.tensor_factory.tensor(temp1)

	res = PondPrivateTensor(prot =w_private.prot,
		share0 = temp0,
		share1 = temp1,
		is_scaled = False)
	res.is_scaled = True
	# with tfe.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	print('res		\n', sess.run(res.reveal()))
	return res

# # test prune
# w_private = prot.define_private_variable(w)
# mask_public = get_weight_mask(w_private, threshold = 0.75, prot = prot)
# weight_prune = w_private * mask_public
# save_private_sparse_matrix(weight_prune)
# weight_prune_load = load_private_sparse_matrix(1,weight_prune.shape)

# with tfe.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	print('w_private	\n', sess.run(w_private.reveal()))
# 	# print('mask_public\n', sess.run(mask_public))
# 	print('weight_prune		 \n', sess.run(weight_prune.reveal()))
# 	print('weight_prune_load		 \n', sess.run(weight_prune_load.reveal()))
