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
from .model_with_crf_loss import CustomModelWithCRFLoss
from evaluation_tools import EvaluationTools

ROOT_PATH = './'
DATA_PATH = ROOT_PATH + 'data/'
TOKENIZER_PATH = ROOT_PATH + 'tokenizers/'

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	from_logits=False, reduction=tf.keras.losses.Reduction.NONE
)

class BaseSingleTask:
	def __init__(self, arch, distil_bert=True, indonlu=True, base_version=True, with_crf=False):
		# Following attributes will be defined on subclass:
			# self.dataset_file
			# self.SPECIAL_TOKEN_LABEL
			# self.task
			# self.tag_set

		self.MODEL_PATH = ROOT_PATH + 'models/{}/{}/'.format(self.task, arch)
		if not os.path.exists(self.MODEL_PATH):
			os.makedirs(self.MODEL_PATH)

		self.EVALUATION_PATH = ROOT_PATH + 'results/{}/{}/'.format(self.task, arch)
		if not os.path.exists(self.EVALUATION_PATH):
			os.makedirs(self.EVALUATION_PATH)

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
		self.with_crf = with_crf

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
		dataset = pd.read_csv(DATA_PATH + self.dataset_file)
		if self.task == 'sentence_recognition':
			dataset.loc[:, 'Tag'] = dataset['IOB'] + '-' + dataset['Type']
			dataset = dataset.drop(columns=['Document ID', 'Sentence #', 'IOB', 'Type'])
		elif self.task == 'boundary_detection':
			dataset.loc[:, 'Tag'] = dataset['IOB']
			dataset = dataset.drop(columns=['Document ID', 'Sentence #', 'IOB', 'Type'])
		self.dataset = dataset

	def _build_tag_encoder(self):
		label = self.dataset['Tag'].unique()
		self.tag_encoder = LabelEncoder().fit(label)
		self.NUM_TAGS = len(label)

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

		df.loc[:, "Tag"] = self.tag_encoder.transform(df["Tag"])

		sentences = df.groupby("Request #")["Word"].apply(list).values
		tag = df.groupby("Request #")["Tag"].apply(list).values
		return sentences, tag

	def create_inputs_targets(self, dataset, indexes):
		dataset_dict = {
			"input_ids": [],
			"attention_mask": [],
			"tags": [],
		}
		sentences, tags = self.process_csv(dataset, indexes)

		for sentence, tag in zip(sentences, tags):
			input_ids = []
			target_tags = []
			for idx, word in enumerate(sentence):
				ids = self.tokenizer.encode(word, add_special_tokens=False)
				input_ids.extend(ids.ids)
				num_tokens = len(ids)
				target_tags.extend([tag[idx]] * num_tokens)

			input_ids = input_ids[:self.MAX_LEN - 2]
			target_tags = target_tags[:self.MAX_LEN - 2]

			input_ids = [self.CLS] + input_ids + [self.SEP]
			attention_mask = [1] * len(input_ids)
			target_tags = [self.SPECIAL_TOKEN_LABEL] + target_tags + [self.SPECIAL_TOKEN_LABEL]

			padding_len = self.MAX_LEN - len(input_ids)

			input_ids = input_ids + ([self.PAD] * padding_len)
			attention_mask = attention_mask + ([0] * padding_len)
			target_tags = target_tags + ([self.SPECIAL_TOKEN_LABEL] * padding_len)

			dataset_dict["input_ids"].append(input_ids)
			dataset_dict["attention_mask"].append(attention_mask)
			dataset_dict["tags"].append(target_tags)

			assert len(target_tags) == self.MAX_LEN, f'{len(input_ids)}, {len(target_tags)}'

		for key in dataset_dict:
			dataset_dict[key] = np.array(dataset_dict[key])

		x = [
			dataset_dict["input_ids"],
			dataset_dict["attention_mask"],
		]
		y = dataset_dict["tags"]
		return x, y

	def masked_loss(self, real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, self.SPECIAL_TOKEN_LABEL))
		loss_ = loss_object(real, pred)

		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		return tf.reduce_mean(loss_)

	def masked_accuracy(self, y_true, y_pred):
		class_id_preds = keras.backend.cast(keras.backend.argmax(y_pred, axis=-1), 'float32')
		ignore_mask = keras.backend.cast(keras.backend.not_equal(y_true, self.SPECIAL_TOKEN_LABEL), 'int32')
		matches = keras.backend.cast(keras.backend.equal(y_true, class_id_preds), 'int32') * ignore_mask
		accuracy = keras.backend.sum(matches) / keras.backend.maximum(keras.backend.sum(ignore_mask), 1)
		return accuracy

	def create_model(self, get_model, train_bert=False):
		input_ids, attention_mask, tag_logits = get_model()

		model = keras.Model(
			inputs=[input_ids, attention_mask],
			outputs=tag_logits,
		)
		
		optimizer = keras.optimizers.Adam(learning_rate=5e-5)
		if self.with_crf:
			model = CustomModelWithCRFLoss(model)
			model.compile(optimizer=optimizer)
		else:
			model.compile(optimizer=optimizer, loss=self.masked_loss, metrics=self.masked_accuracy)	

		for layer in self.bert_model.layers:
			layer.trainable = train_bert

		return model

	def get_prediction(self, model, index, label_dict, file_path):
		x_test, y_test = self.create_inputs_targets(self.dataset, indexes=[index])

		pred_tags = None
		if self.with_crf:
			pred_tags = model.predict(x_test)[0]
			pred_tags = np.insert(pred_tags, 0, self.SPECIAL_TOKEN_LABEL)
		else:
			pred_test = model.predict(x_test)
			pred_tags = np.argmax(pred_test, 2)[0]

		mask = x_test[1][0]
		assert len(pred_tags) == len(mask)

		golds = [label_dict[tag] for tag in y_test[0] if tag != self.SPECIAL_TOKEN_LABEL]
		pred_tags = [pred_tags[i] for i in range(len(pred_tags)) if mask[i] != 0][1:-1]
		predictions = [label_dict[pred_tags[i]] for i in range(len(pred_tags))]
		assert len(golds) == len(predictions)

		ids = x_test[0][0]
		assert len(ids) == len(mask)

		tokens = [self.tokenizer.id_to_token(ids[i]) for i in range(len(ids)) if mask[i] != 0][1:-1]
		assert len(tokens) == len(golds)

		with open(self.EVALUATION_PATH + file_path, 'a') as file_object:
			file_object.write('Index: {}\n'.format(index))
			for i in range(len(golds)):
				file_object.write('{} {} {}\n'.format(tokens[i], golds[i], predictions[i]))
			file_object.write('\n')

		return golds, predictions

	def evaluate_model(
			self,
			get_model,
			folds=[i for i in range(1, 11)],
			train_bert=False,
			epochs=20,
			batch_size=32,
			verbose=1,
			save_model=False
		):

		kf = KFold(n_splits=10)
		golds = []
		predictions = []
		test_set_indexes = []

		fold_index = 1
		for (train_indexes, test_indexes) in kf.split(self.dataset['Request #'].unique()):
			if fold_index not in folds:
				fold_index += 1
				continue

			test_set_indexes += list(test_indexes)
			x_train, y_train = self.create_inputs_targets(self.dataset, indexes=train_indexes)

			model = self.create_model(get_model, train_bert)
			model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

			if save_model:
				model.save(self.MODEL_PATH + 'fold_{}.h5'.format(fold_index), save_format='tf')

			le_dict = dict(zip(self.tag_encoder.transform(self.tag_encoder.classes_), self.tag_encoder.classes_))

			for test_index in test_indexes:
				gold, prediction = self.get_prediction(model, test_index, le_dict, 'fold_{}.txt'.format(fold_index))
				golds += gold
				predictions += prediction

			print('Fold Done: {}'.format(fold_index))
			fold_index += 1

		print()
		print(classification_report(golds, predictions, labels=self.tag_set, digits=4))
		print('\n{}'.format(self.tag_set))
		print(confusion_matrix(golds, predictions, labels=self.tag_set))

	def post_evaluate(self, version=1, folds=[i for i in range(1, 11)]):
		eval_tools = EvaluationTools(self.EVALUATION_PATH)
		eval_tools.post_evaluation(self.task, version, folds)
