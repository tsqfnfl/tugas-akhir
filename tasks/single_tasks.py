from .base_single_task import BaseSingleTask

class SentenceRecognition(BaseSingleTask):
	def __init__(self, arch, distil_bert=True, indonlu=True, base_version=True, with_crf=False):
		self.dataset_file = 'sr_dataset.csv'
		self.SPECIAL_TOKEN_LABEL = 6
		self.task = 'sentence_recognition'
		self.tag_set = ['B-BACKGROUND', 'I-BACKGROUND', 'B-QUESTION', 'I-QUESTION', 'B-IGNORE', 'I-IGNORE']
		super().__init__(arch, distil_bert, indonlu, base_version, with_crf)

class BoundaryDetection(BaseSingleTask):
	def __init__(self, arch, distil_bert=True, indonlu=True, base_version=True, with_crf=False):
		self.dataset_file = 'sr_dataset.csv'
		self.SPECIAL_TOKEN_LABEL = 2
		self.task = 'boundary_detection'
		self.tag_set = ['B', 'I']
		super().__init__(arch, distil_bert, indonlu, base_version, with_crf)

class MER(BaseSingleTask):
	def __init__(self, arch, distil_bert=True, indonlu=True, base_version=True, with_crf=False):
		self.dataset_file = 'mer_dataset.csv'
		self.SPECIAL_TOKEN_LABEL = 9
		self.task = 'medical_entity_recognition'
		self.tag_set = ['B-Disease', 'B-Drug', 'B-Symptom', 'B-Treatment', 'I-Disease', 'I-Drug', 'I-Symptom', 'I-Treatment', 'OO']
		super().__init__(arch, distil_bert, indonlu, base_version, with_crf)


class KeyphrasesExtraction(BaseSingleTask):
	def __init__(self, arch, distil_bert=True, indonlu=True, base_version=True, with_crf=False):
		self.dataset_file = 'ke_dataset.csv'
		self.SPECIAL_TOKEN_LABEL = 3
		self.task = 'keyphrases_extraction'
		self.tag_set = ['BK', 'IK', 'OO']
		super().__init__(arch, distil_bert, indonlu, base_version, with_crf)