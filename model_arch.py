import random
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

from tensorflow.keras import layers
from tf2crf import CRF
from keras.models import Sequential

class CustomCRF(CRF):
	def call(self, inputs, mask=None, training=False):
		if mask is None:
			raw_input_shape = tf.slice(tf.shape(inputs), [0], [2])
			mask = tf.ones(raw_input_shape)
		sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1) 
		if self.units:
			inputs = self.dense(inputs)[:, 1:, :]
		sequence_lengths = tf.add(sequence_lengths, -2)
		viterbi_sequence, _ = tfa.text.crf_decode(
			inputs, self.transitions, sequence_lengths
		)
		return viterbi_sequence, inputs, sequence_lengths, self.transitions

def tf_seed(seed=0):
	os.environ['PYTHONHASHSEED'] = str(seed)
	os.environ["TF_DETERMINISTIC_OPS"] = str(seed)
	os.environ['CUDA_VISBLE_DEVICE'] = ''
	np.random.seed(seed)
	random.seed(seed)
	tf.random.set_seed(seed)


def model_dense(task):
	tf_seed(seed=0)

	input_ids = layers.Input(shape=(task.MAX_LEN,), dtype=tf.int32)
	attention_mask = layers.Input(shape=(task.MAX_LEN,), dtype=tf.int32)

	task.load_bert_model()
	embedding = task.bert_model(
		input_ids, attention_mask=attention_mask
	)[0]

	tag_logits = layers.TimeDistributed(
		layers.Dense(
			task.NUM_TAGS + 1,
			activation='softmax',
		)
	)(embedding)

	return input_ids, attention_mask, tag_logits

def model_bilstm(task):
	tf_seed(seed=0)

	input_ids = layers.Input(shape=(task.MAX_LEN,), dtype=tf.int32)
	attention_mask = layers.Input(shape=(task.MAX_LEN,), dtype=tf.int32)

	task.load_bert_model()
	embedding = task.bert_model(
		input_ids, attention_mask=attention_mask
	)[0]
	bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(embedding)

	tag_logits = layers.TimeDistributed(
		layers.Dense(
			task.NUM_TAGS + 1,
			activation='softmax',
		)
	)(bilstm)

	return input_ids, attention_mask, tag_logits

def model_bilstm_crf(task):
	tf_seed(seed=0)

	input_ids = layers.Input(shape=(task.MAX_LEN,), dtype=tf.int32)
	attention_mask = layers.Input(shape=(task.MAX_LEN,), dtype=tf.int32)

	task.load_bert_model()
	embedding = task.bert_model(
		input_ids, attention_mask=attention_mask
	)[0]
	bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(embedding)
	crf = CustomCRF(task.NUM_TAGS + 1)
	output = crf(bilstm, mask=attention_mask)

	return input_ids, attention_mask, output

def model_lstm_for_sc(max_length, num_tags):
    tf_seed(seed=0)

    model = Sequential()
    model.add(layers.Masking(mask_value=0., input_shape=(max_length, 768)))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(num_tags, activation="softmax")))
    return model