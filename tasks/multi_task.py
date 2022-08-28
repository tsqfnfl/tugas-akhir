import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, initializers
from tokenizers import BertWordPieceTokenizer
from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import AutoTokenizer, TFAutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from evaluation_tools import EvaluationTools

ROOT_PATH = './'
DATA_PATH = ROOT_PATH + 'data/multi_task/'
TOKENIZER_PATH = ROOT_PATH + 'tokenizers/'

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	from_logits=False, reduction=tf.keras.losses.Reduction.NONE
)

class MultiTask:
	def __init__(self, arch, distil_bert=True, indonlu=True, base_version=True):
		self.dataset_file = 'multi_task_dataset.csv'

		self.KP_SPECIAL_LABEL = 3
		self.MER_SPECIAL_LABEL = 9
		
		self.kp_tag_set = ['BK', 'IK', 'OO']
		self.mer_tag_set = ['B-Disease', 'B-Drug', 'B-Symptom', 'B-Treatment', 'I-Disease', 'I-Drug', 'I-Symptom', 'I-Treatment', 'OO']

		self.MODEL_PATH = ROOT_PATH + 'models/multi_task/{}/'.format(arch)
		if not os.path.exists(self.MODEL_PATH):
			os.makedirs(self.MODEL_PATH)

		self.EVALUATION_PATH = ROOT_PATH + 'results/multi_task/{}/'.format(arch)
		self.KP_EVALUATION_PATH = self.EVALUATION_PATH + 'keyphrases_extraction/'
		self.MER_EVALUATION_PATH = self.EVALUATION_PATH + 'medical_entity_recognition/'
		if not os.path.exists(self.EVALUATION_PATH):
			os.makedirs(self.KP_EVALUATION_PATH)
			os.makedirs(self.MER_EVALUATION_PATH)

		self.MAX_LEN = 450

		self._load_dataset()
		self._build_tag_encoder()

		if distil_bert:
			self.bert_model_name = 'cahya/distilbert-base-indonesian'
		elif indonlu:
			if base_version:
				self.bert_model_name = 'indobenchmark/indobert-base-p1'
			else:
				self.bert_model_name = 'indobenchmark/indobert-large-p1'
		else:
			self.bert_model_name = 'indolem/indobert-base-uncased'

		self.distil_bert = distil_bert
		self.indonlu = indonlu
		self.base_version = base_version

		print('Load tokenizer...')
		self._load_tokenizer()
		self._compute_max_length()

		print('\nLoad BERT model...')
		self.load_bert_model()

		if distil_bert:
			self.CLS = 3
			self.SEP = 1
			self.PAD = 2
		else:
			self.PAD = 0
			self.CLS = 2 if indonlu else 3
			self.SEP = 3 if indonlu else 4

	def _load_dataset(self):
		self.dataset = pd.read_csv(DATA_PATH + self.dataset_file)

	def _build_tag_encoder(self):
		keyphrase_label = self.dataset['Keyphrase'].unique()
		self.keyphrase_tag_encoder = LabelEncoder().fit(keyphrase_label)
		self.KEYPHRASE_NUM_TAGS = len(keyphrase_label)

		mer_label = self.dataset['MER'].unique()
		self.mer_tag_encoder = LabelEncoder().fit(mer_label)
		self.MER_NUM_TAGS = len(mer_label)

	def _load_tokenizer(self):
		if self.distil_bert:
			tokenizer_path = TOKENIZER_PATH + 'tokenizer_distilbert_base_indonesian/'
			self.slow_tokenizer = DistilBertTokenizer.from_pretrained(self.bert_model_name)
			try:
				self.tokenizer = BertWordPieceTokenizer(tokenizer_path + 'vocab.txt', lowercase=True)
			except:
				if not os.path.exists(tokenizer_path):
					os.makedirs(tokenizer_path)
				self.slow_tokenizer.save_pretrained(tokenizer_path)
				self.tokenizer = BertWordPieceTokenizer(tokenizer_path + 'vocab.txt', lowercase=True)

		else:
			tokenizer_path = None
			if self.indonlu:
				if self.base_version:
					tokenizer_path = TOKENIZER_PATH + 'tokenizer_indobert_base_p1/'
				else:
					tokenizer_path = TOKENIZER_PATH + 'tokenizer_indobert_large_p1/'
			else:
				tokenizer_path = TOKENIZER_PATH + 'tokenizer_indobert_base_uncased/'
			
			self.slow_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
			try:
				self.tokenizer = BertWordPieceTokenizer(tokenizer_path + 'vocab.txt', lowercase=True)
			except:
				if not os.path.exists(tokenizer_path):
					os.makedirs(tokenizer_path)
				self.slow_tokenizer.save_pretrained(tokenizer_path)
				self.tokenizer = BertWordPieceTokenizer(tokenizer_path + 'vocab.txt', lowercase=True)


	def _compute_max_length(self):
		sentences = list(self.dataset.groupby('Request #').agg({'Word': ' '.join})['Word'])
		encoded_sentences = self.slow_tokenizer(sentences, padding='longest')
		sentence_max_length = len(encoded_sentences['input_ids'][0])
		del self.slow_tokenizer
		
		try:
			assert sentence_max_length <= self.MAX_LEN
		except:
			print('Warning: Longest sentence in the dataset is more than MAX_LEN, some input sentences will be truncated!')

	def load_bert_model(self):
		if self.distil_bert:
			self.bert_model = TFDistilBertModel.from_pretrained(self.bert_model_name, from_pt=True)
		else:
			self.bert_model = TFAutoModel.from_pretrained(self.bert_model_name, from_pt=True)

	def process_csv(self, dataset, indexes):
		selected_index = ['Request: {}'.format(i) for i in indexes]
		df = dataset.loc[dataset['Request #'].isin(selected_index)].copy()

		df.loc[:, "Keyphrase"] = self.keyphrase_tag_encoder.transform(df["Keyphrase"])
		df.loc[:, "MER"] = self.mer_tag_encoder.transform(df["MER"])

		sentences = df.groupby("Request #")["Word"].apply(list).values
		keyphrase_tag = df.groupby("Request #")["Keyphrase"].apply(list).values
		mer_tag = df.groupby("Request #")["MER"].apply(list).values
		return sentences, keyphrase_tag, mer_tag

	def create_inputs_targets(self, dataset, indexes):
		dataset_dict = {
			"input_ids": [],
			"attention_mask": [],
			"keyphrase_tags": [],
			"mer_tags": [],
		}
		sentences, keyphrase_tags, mer_tags = self.process_csv(dataset, indexes)

		for sentence, keyphrase_tag, mer_tag in zip(sentences, keyphrase_tags, mer_tags):
			input_ids = []
			keyphrase_target_tags = []
			mer_target_tags = []
			for idx, word in enumerate(sentence):
				ids = self.tokenizer.encode(word, add_special_tokens=False)
				input_ids.extend(ids.ids)
				num_tokens = len(ids)
				keyphrase_target_tags.extend([keyphrase_tag[idx]] * num_tokens)
				mer_target_tags.extend([mer_tag[idx]] * num_tokens)

			input_ids = input_ids[:self.MAX_LEN - 2]
			keyphrase_target_tags = keyphrase_target_tags[:self.MAX_LEN - 2]
			mer_target_tags = mer_target_tags[:self.MAX_LEN - 2]

			input_ids = [self.CLS] + input_ids + [self.SEP]
			attention_mask = [1] * len(input_ids)
			keyphrase_target_tags = [self.KP_SPECIAL_LABEL] + keyphrase_target_tags + [self.KP_SPECIAL_LABEL]
			mer_target_tags = [self.MER_SPECIAL_LABEL] + mer_target_tags + [self.MER_SPECIAL_LABEL]

			padding_len = self.MAX_LEN - len(input_ids)

			input_ids = input_ids + ([self.PAD] * padding_len)
			attention_mask = attention_mask + ([0] * padding_len)
			keyphrase_target_tags = keyphrase_target_tags + ([self.KP_SPECIAL_LABEL] * padding_len)
			mer_target_tags = mer_target_tags + ([self.MER_SPECIAL_LABEL] * padding_len)

			dataset_dict["input_ids"].append(input_ids)
			dataset_dict["attention_mask"].append(attention_mask)
			dataset_dict["keyphrase_tags"].append(keyphrase_target_tags)
			dataset_dict["mer_tags"].append(mer_target_tags)

			assert len(keyphrase_target_tags) == self.MAX_LEN, f'{len(input_ids)}, {len(keyphrase_target_tags)}'
			assert len(mer_target_tags) == self.MAX_LEN, f'{len(input_ids)}, {len(mer_target_tags)}'

		for key in dataset_dict:
			dataset_dict[key] = np.array(dataset_dict[key])

		x = [
			dataset_dict["input_ids"],
			dataset_dict["attention_mask"],
		]
		y = [
			dataset_dict["keyphrase_tags"],
			dataset_dict["mer_tags"],
		]
		return x, y

	def masked_keyphrase_loss(self, real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, self.KP_SPECIAL_LABEL))
		loss_ = loss_object(real, pred)

		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		return tf.reduce_mean(loss_)

	def masked_mer_loss(self, real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, self.MER_SPECIAL_LABEL))
		loss_ = loss_object(real, pred)

		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		return tf.reduce_mean(loss_)

	def masked_keyphrase_accuracy(self, y_true, y_pred):
		class_id_preds = keras.backend.cast(keras.backend.argmax(y_pred, axis=-1), 'float32')
		ignore_mask = keras.backend.cast(keras.backend.not_equal(y_true, self.KP_SPECIAL_LABEL), 'int32')
		matches = keras.backend.cast(keras.backend.equal(y_true, class_id_preds), 'int32') * ignore_mask
		accuracy = keras.backend.sum(matches) / keras.backend.maximum(keras.backend.sum(ignore_mask), 1)
		return accuracy

	def masked_mer_accuracy(self, y_true, y_pred):
		class_id_preds = keras.backend.cast(keras.backend.argmax(y_pred, axis=-1), 'float32')
		ignore_mask = keras.backend.cast(keras.backend.not_equal(y_true, self.MER_SPECIAL_LABEL), 'int32')
		matches = keras.backend.cast(keras.backend.equal(y_true, class_id_preds), 'int32') * ignore_mask
		accuracy = keras.backend.sum(matches) / keras.backend.maximum(keras.backend.sum(ignore_mask), 1)
		return accuracy

	def create_model(self, get_model, alpha=0.5, beta=None, train_bert=False):
		input_ids, attention_mask, keyphrase_tag_logits, mer_tag_logits = get_model()

		model = keras.Model(
			inputs=[input_ids, attention_mask],
			outputs=[keyphrase_tag_logits, mer_tag_logits],
		)
		optimizer = keras.optimizers.Adam(learning_rate=5e-5)
		model.compile(
			optimizer=optimizer,
			loss={'keyphrase': self.masked_keyphrase_loss, 'mer': self.masked_mer_loss},
			loss_weights={'keyphrase': alpha, 'mer': beta if beta is not None else (1 - alpha)},
			metrics={'keyphrase': self.masked_keyphrase_accuracy, 'mer': self.masked_mer_accuracy}
		)

		for layer in self.bert_model.layers:
			layer.trainable = train_bert

		return model

	def get_prediction(self, model, index, keyphrase_label_dict, mer_label_dict, file_path):
		x_test, y_test = self.create_inputs_targets(self.dataset, indexes=[index])
		pred_test = model.predict(x_test)
		keyphrase_pred_tags = np.argmax(pred_test[0], 2)[0]
		mer_pred_tags = np.argmax(pred_test[1], 2)[0]

		mask = x_test[1][0]
		assert len(keyphrase_pred_tags) == len(mask)
		assert len(mer_pred_tags) == len(mask)

		keyphrase_golds = [keyphrase_label_dict[tag] for tag in y_test[0][0] if tag != self.KP_SPECIAL_LABEL]
		keyphrase_predictions = [keyphrase_label_dict[keyphrase_pred_tags[i]] for i in range(len(keyphrase_pred_tags)) if mask[i] != 0][1:-1]
		assert len(keyphrase_golds) == len(keyphrase_predictions)

		mer_golds = [mer_label_dict[tag] for tag in y_test[1][0] if tag != self.MER_SPECIAL_LABEL]
		mer_predictions = [mer_label_dict[mer_pred_tags[i]] for i in range(len(mer_pred_tags)) if mask[i] != 0][1:-1]
		assert len(mer_golds) == len(mer_predictions)

		ids = x_test[0][0]
		assert len(ids) == len(mask)

		tokens = [self.tokenizer.id_to_token(ids[i]) for i in range(len(ids)) if mask[i] != 0][1:-1]
		assert len(tokens) == len(keyphrase_golds)
		assert len(tokens) == len(mer_golds)

		with open(self.KP_EVALUATION_PATH + file_path, 'a') as file_object:
			file_object.write('Index: {}\n'.format(index))
			for i in range(len(keyphrase_golds)):
				file_object.write('{} {} {}\n'.format(tokens[i], keyphrase_golds[i], keyphrase_predictions[i]))
			file_object.write('\n')

		with open(self.MER_EVALUATION_PATH + file_path, 'a') as file_object:
			file_object.write('Index: {}\n'.format(index))
			for i in range(len(mer_golds)):
				file_object.write('{} {} {}\n'.format(tokens[i], mer_golds[i], mer_predictions[i]))
			file_object.write('\n')

		return keyphrase_golds, keyphrase_predictions, mer_golds, mer_predictions

	def evaluate_model(
			self,
			get_model,
			folds=[i for i in range(1, 11)],
			alpha=0.5,
			beta=None,
			train_bert=False,
			epochs=20,
			batch_size=32,
			verbose=1,
			save_model=False
		):

		kf = KFold(n_splits=10)
		kp_golds = []
		kp_predictions = []
		mer_golds = []
		mer_predictions = []
		test_set_indexes = []

		if beta is None:
			beta = 1 - alpha

		fold_index = 1
		for (train_indexes, test_indexes) in kf.split(self.dataset['Request #'].unique()):
			if fold_index not in folds:
				fold_index += 1
				continue

			test_set_indexes += list(test_indexes)
			x_train, y_train = self.create_inputs_targets(self.dataset, indexes=train_indexes)

			model = self.create_model(get_model, alpha, beta, train_bert)
			model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

			if save_model:
				model.save(self.MODEL_PATH + 'fold_{}.h5'.format(fold_index), save_format='tf')

			keyphrase_le_dict = dict(zip(self.keyphrase_tag_encoder.transform(self.keyphrase_tag_encoder.classes_), self.keyphrase_tag_encoder.classes_))
			mer_le_dict = dict(zip(self.mer_tag_encoder.transform(self.mer_tag_encoder.classes_), self.mer_tag_encoder.classes_))

			for test_index in test_indexes:
				kp_gold, kp_prediction, mer_gold, mer_prediction = self.get_prediction(model, test_index, keyphrase_le_dict, mer_le_dict, 'fold_{}.txt'.format(fold_index))
				kp_golds += kp_gold
				kp_predictions += kp_prediction
				mer_golds += mer_gold
				mer_predictions += mer_prediction

			print('Fold Done: {}'.format(fold_index))
			fold_index += 1

		print()
		print('===== Keyphrases Extraction =====')
		print(classification_report(kp_golds, kp_predictions, labels=self.kp_tag_set))
		print('\n{}'.format(self.kp_tag_set))
		print(confusion_matrix(kp_golds, kp_predictions, labels=self.kp_tag_set))

		print()
		print('===== Medical Entity Recognition =====')
		print(classification_report(mer_golds, mer_predictions, labels=self.mer_tag_set))
		print('\n{}'.format(self.mer_tag_set))
		print(confusion_matrix(mer_golds, mer_predictions, labels=self.mer_tag_set))

	def post_evaluate(self, task, version=1, folds=[i for i in range(1, 11)]):
		if task == 'keyphrases_extraction':
			evaluation_path = self.KP_EVALUATION_PATH
		elif task == 'medical_entity_recognition':
			evaluation_path = self.MER_EVALUATION_PATH
		eval_tools = EvaluationTools(evaluation_path, multi_task=True)
		eval_tools.post_evaluation(task, version, folds)
