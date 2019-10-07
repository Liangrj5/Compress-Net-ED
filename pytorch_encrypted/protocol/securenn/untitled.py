import torch

Factory 
NATIVE_TYPE = tf.int32
x = Factory.tensor(tf.constant([2, -2], dtype=tf.int64))


	class Factory:

		def tensor(self, value):

			if isinstance(value, tf.Tensor):
				if value.dtype is not NATIVE_TYPE:
					value = tf.cast(value, dtype=NATIVE_TYPE)
				# no assumptions are made about the tensor here and hence we need to
				# apply our mapping for invalid values
				value = _map_minusone_to_zero(value, NATIVE_TYPE)
				return OddDenseTensor(value)

			raise TypeError("Don't know how to handle {}".format(type(value)))

		def constant(self, value):
			raise NotImplementedError()

		def variable(self, initial_value):
			raise NotImplementedError()

		def placeholder(self, shape):
			raise NotImplementedError()

		@property
		def modulus(self):

			if NATIVE_TYPE is tf.int32:
				return 2**32 - 1

			if NATIVE_TYPE is tf.int64:
				return 2**64 - 1

			raise NotImplementedError(("Incorrect native type ",
																 "{}.".format(NATIVE_TYPE)))

		@property
		def native_type(self):
			return NATIVE_TYPE

		def sample_uniform(self, shape, minval: Optional[int] = None, maxval: Optional[int] = None):
			"""Sample a tensor from a uniform distribution."""
			assert minval is None
			assert maxval is None

			if secure_random.supports_seeded_randomness():
				seed = secure_random.secure_seed()
				return OddUniformTensor(shape=shape, seed=seed)

			if secure_random.supports_secure_randomness():
				sampler = secure_random.random_uniform
			else:
				sampler = tf.random_uniform

			value = _construct_value_from_sampler(sampler=sampler, shape=shape)
			return OddDenseTensor(value)

		def sample_bounded(self, shape, bitlength: int):
			raise NotImplementedError()

		def stack(self, xs: list, axis: int = 0):
			raise NotImplementedError()

		def concat(self, xs: list, axis: int):
			raise NotImplementedError()

	master_factory = Factory()


	class OddTensor(AbstractTensor):
		"""
		Base class for the concrete odd tensors types.

		Implements basic functionality needed by SecureNN subprotocols from a few
		abstract properties implemented by concrete types below.
		"""

		@property
		def factory(self):
			return master_factory

		@property
		@abc.abstractproperty
		def value(self) -> torch.Tensor:
			pass

		@property
		@abc.abstractproperty
		def shape(self):
			pass

		def identity(self):
			value = self.value.clone()
			return OddDenseTensor(value)

		def clone(self):
			value = self.value.clone()
			return OddDenseTensor(value)


		def __repr__(self) -> str:
			return '{}(shape={}, NATIVE_TYPE={})'.format(
					type(self),
					self.shape,
					NATIVE_TYPE,
			)

		def __getitem__(self, slc):
			return OddDenseTensor(self.value[slc])

		def __add__(self, other):
			return self.add(other)

		def __sub__(self, other):
			return self.sub(other)

		def add(self, other):
			"""Add other to this tensor."""
			x, y = _lift(self, other)
			bitlength = math.ceil(math.log2(master_factory.modulus))

			with tf.name_scope('add'):

				# the below avoids redundant seed expansion; can be removed once
				# we have a (per-device) caching mechanism in place
				x_value = x.value
				y_value = y.value

				z = x_value + y_value

				with tf.name_scope('correct_wrap'):

					# we want to compute whether we wrapped around, ie `pos(x) + pos(y) >= m - 1`,
					# for correction purposes which, since `m - 1 == 1` for signed integers, can be
					# rewritten as:
					#	-> `pos(x) >= m - 1 - pos(y)`
					#	-> `m - 1 - pos(y) - 1 < pos(x)`
					#	-> `-1 - pos(y) - 1 < pos(x)`
					#	-> `-2 - pos(y) < pos(x)`
					wrapped_around = _lessthan_as_unsigned(-2 - y_value,
																								 x_value,
																								 bitlength)
					z += wrapped_around

			return OddDenseTensor(z)

	class OddDenseTensor(OddTensor):
		"""
		Represents a tensor with explicit values, as opposed to OddUniformTensor
		with implicit values.

		Internal use only and assume that invalid values have already been mapped.
		"""

		def __init__(self, value):
			assert isinstance(value, torch.Tensor)
			self._value = value

		@property
		def value(self) -> torch.Tensor:
			return self._value

		@property
		def shape(self):
			return self._value.shape
				
				def size(self, dim = None):
						if dim is None:
								return self.size()
						else:
								return self.size(dim)

		@property
		def support(self):
			return [self._value]

	def _lift(x, y) -> Tuple[OddTensor, OddTensor]:
		"""
		Attempts to lift x and y to compatible OddTensors for further processing.
		"""

		if isinstance(x, OddTensor) and isinstance(y, OddTensor):
			assert x.factory == y.factory, "Incompatible types: {} and {}".format(
					x.factory, y.factory)
			return x, y

		if isinstance(x, OddTensor):

			if isinstance(y, int):
				return x, x.factory.tensor(np.array([y]))

		if isinstance(y, OddTensor):

			if isinstance(x, int):
				return y.factory.tensor(np.array([x])), y

		raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))
