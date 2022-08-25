import tensorflow as tf
from tf2crf import ModelWithCRFLoss

def unpack_data(data):
	data = list(data)
	data[1] = data[1][:, 1:]
	data = tuple(data)
	if len(data) == 2:
		return data[0], data[1], None
	elif len(data) == 3:
		return data
	else:
		raise TypeError("Expected data to be a tuple of size 2 or 3.")

class CustomModelWithCRFLoss(ModelWithCRFLoss):
	def train_step(self, data):
		x, y, sample_weight = unpack_data(data)
		if self.sparse_target:
			assert len(y.shape) == 2
		else:
			y = tf.argmax(y, axis=-1)
		with tf.GradientTape() as tape:
			viterbi_sequence, sequence_length, crf_loss = self.compute_loss(x, y, training=True)
			loss = crf_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		self.loss_tracker.update_state(loss)
		self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
		return {"loss": self.loss_tracker.result(), self.metrics_fn.name: self.metrics_fn.result()}

	def test_step(self, data):
		x, y, sample_weight = unpack_data(data)
		if self.sparse_target:
			assert len(y.shape) == 2
		else:
			y = tf.argmax(y, axis=-1)
		viterbi_sequence, sequence_length, crf_loss = self.compute_loss(x, y, training=True)
		loss = crf_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
		self.loss_tracker.update_state(loss)
		self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
		return {"loss_val": self.loss_tracker.result(), f'val_{self.metrics_fn.name}': self.metrics_fn.result()}