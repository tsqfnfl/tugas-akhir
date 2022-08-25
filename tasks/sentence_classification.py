import os
import numpy as np
import pandas as pd

from tokenizers import BertWordPieceTokenizer
from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import AutoTokenizer, TFAutoModel
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from keras.preprocessing.sequence import pad_sequences

ROOT_PATH = './'
DATA_PATH = ROOT_PATH + 'data/'

class SentenceClassification:
	def __init__(self, distil_bert=True, indonlu=True, base_version=True):
		self.DATASET_PATH = DATA_PATH + 'sc_dataset.csv'
		if not os.path.exists(self.DATASET_PATH):
			self._generate_dataset()

		self.MODEL_PATH = ROOT_PATH + 'models/sentence_classification/'
		if not os.path.exists(self.MODEL_PATH):
			os.makedirs(self.MODEL_PATH)

		self.EVALUATION_PATH = ROOT_PATH + 'results/sentence_classification/'
		if not os.path.exists(self.EVALUATION_PATH):
			os.makedirs(self.EVALUATION_PATH)

		self.MAX_LEN = 40

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


	def _generate_dataset(self):
		dataset = pd.read_csv(DATA_PATH + 'sr_dataset.csv')
		sentence_dataset = dataset.groupby('Sentence #', as_index=False).agg({'Request #': 'first', 'Document ID': 'first', 'Word': ' '.join, 'Type': 'first'})

		sentence_dataset['request_num'] = sentence_dataset['Request #'].apply(lambda x: int(x.split(': ')[-1]))
		sentence_dataset['sentence_num'] = sentence_dataset['Sentence #'].apply(lambda x: int(x.split(': ')[-1]))

		sentence_dataset = sentence_dataset.sort_values(by=['request_num', 'sentence_num']).reset_index()
		sentence_dataset = sentence_dataset.drop(columns=['index', 'request_num', 'sentence_num', 'Document ID'])
		sentence_dataset = sentence_dataset[['Request #', 'Sentence #', 'Word', 'Type']]
		sentence_dataset.to_csv(self.DATASET_PATH, index=False)

	def _load_dataset(self):
		self.dataset = pd.read_csv(self.DATASET_PATH)

	def _build_tag_encoder(self):
		label = self.dataset['Type'].unique()
		self.tag_encoder = LabelBinarizer().fit(label)
		self.NUM_TAGS = len(label)

	def _load_tokenizer(self):
		if self.distil_bert:
			self.tokenizer = DistilBertTokenizer.from_pretrained(self.bert_model_name)
		else:
			self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)

	def _compute_max_length(self):
		max_sentence_count = max(self.dataset.groupby('Request #').size())
		
		try:
			assert max_sentence_count <= self.MAX_LEN
		except:
			print('Warning: Maximum sentence count for a request in the dataset is more than MAX_LEN, some input will be truncated!')

	def load_bert_model(self):
		if self.distil_bert:
			self.bert_model = TFDistilBertModel.from_pretrained(self.bert_model_name, from_pt=True)
		else:
			self.bert_model = TFAutoModel.from_pretrained(self.bert_model_name, from_pt=True)

	def get_predictions(self, model, test_data, fold):
	    x_test = []
	    for index, row in test_data.iterrows():
	        encoded_sentence = self.tokenizer(row['Word'], return_tensors='tf')
	        embedding = self.bert_model(encoded_sentence)
	        cls_out = embedding.last_hidden_state[:, 0, :][0]
	        x_test.append(cls_out)
	    
	    x_test = pad_sequences([x_test], dtype='float32', maxlen=self.MAX_LEN, padding='post')
	    
	    predictions = model.predict(x_test)[0]
	    predictions = np.argmax([list(predictions[i]) for i in range(len(test_data))], axis=-1)
	    predictions = [self.tag_encoder.classes_[x] for x in predictions]

	    with open(self.EVALUATION_PATH + 'fold_{}.txt'.format(fold), 'a') as file_object:
	        assert test_data['Request #'].nunique() == 1
	        index = test_data.iloc[0]['Request #'].split(': ')[-1]

	        file_object.write('Index: {}\n'.format(index))
	        for i in range(len(test_data)):
	            sentence = test_data.iloc[i]
	            file_object.write('{} {} {}\n'.format(sentence['Word'], sentence['Type'], predictions[i]))
	        file_object.write('\n')

	    return list(test_data['Type']), predictions

	def evaluate_model(
		self,
	    get_model,
	    folds=[i for i in range(1, 11)],
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
	        x_train, y_train = [], []

	        for index in train_indexes:
	            x_request = []
	            request = self.dataset[self.dataset['Request #'] == 'Request: {}'.format(index)]
	            for index, row in request.iterrows():
	                encoded_sentence = self.tokenizer(row['Word'], return_tensors='tf')
	                embedding = self.bert_model(encoded_sentence)
	                cls_out = embedding.last_hidden_state[:, 0, :][0]
	                x_request.append(cls_out)
	            x_train.append(x_request)
	            y_train.append(self.tag_encoder.transform(list(request['Type'])))
	        
	        x_train = pad_sequences(x_train, dtype='float32', maxlen=self.MAX_LEN, padding='post')
	        y_train = pad_sequences(y_train, maxlen=self.MAX_LEN, padding='post')

	        model = get_model(self.MAX_LEN, self.NUM_TAGS)
	        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

	        if save_model:
	            model.save(self.MODEL_PATH + 'fold_{}.h5'.format(fold_index), save_format='tf')

	        for test_index in test_indexes:
	            request = self.dataset[self.dataset['Request #'] == 'Request: {}'.format(test_index)]
	            test_golds, test_predictions = self.get_predictions(model, request, fold_index)
	            golds += test_golds
	            predictions += test_predictions
	    
	        print('Fold Done: {}'.format(fold_index))
	        fold_index += 1

	    tag_set = ['BACKGROUND', 'IGNORE', 'QUESTION']

	    print()
	    print(classification_report(golds, predictions, labels=tag_set, digits=4))
	    print('\n{}'.format(tag_set))
	    print(confusion_matrix(golds, predictions, labels=tag_set))
