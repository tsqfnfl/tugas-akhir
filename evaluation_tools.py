import json
import os
from sklearn.metrics import classification_report, confusion_matrix

ROOT_PATH = './'
DATA_PATH = ROOT_PATH + 'data/'

class Evaluator:
    def __init__(self, path, task, folds, multi_task=False):
        self.path = path
        self.task = task
        self.folds = folds

        if not multi_task:
            self.DATA_PATH = DATA_PATH + 'single_task/'
        else:
            self.DATA_PATH = DATA_PATH + 'multi_task/'

    def get_measurement_partial(self, model_phrase, original_phrases):
        model_words = model_phrase.split()
        temp_ori = original_phrases
        for model_word in model_words:
            for original_phrase in original_phrases:
                if model_word in original_phrase.split():
                    temp_ori.remove(original_phrase)
                    return 1, temp_ori
        return 0, temp_ori

    def get_measurement_full(self, model_phrase, original_phrases):
        temp_ori = original_phrases
        for original_phrase in original_phrases:
            if model_phrase == original_phrase:
                temp_ori.remove(original_phrase)
                return 1, temp_ori
        return 0, temp_ori

    def partial_evaluator(self, model_phrases, original_phrases):
        model_phrases = map(lambda x: x.lower().strip(), model_phrases)
        model_phrases = list(set(model_phrases))
        original_phrases = map(lambda x: x.lower().strip(),original_phrases)
        original_phrases = list(set(original_phrases))
        new_ori = original_phrases
        result = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0}
        for model_phrase in model_phrases:
            measurement, temp_ori = self.get_measurement_partial(model_phrase, new_ori)
            new_ori = temp_ori
            if measurement == 1:
                result['tp'] += 1
            else:
                result['fp'] += 1
            if not new_ori:
                break
        result['fn'] = len(new_ori)
        return result

    def full_evaluator(self, model_phrases, original_phrases):
        model_phrases = map(lambda x: x.lower().strip(), model_phrases)
        model_phrases = list(set(model_phrases))
        original_phrases = map(lambda x: x.lower().strip(),original_phrases)
        original_phrases = list(set(original_phrases))
        new_ori = original_phrases
        result = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0}
        for model_phrase in model_phrases:
            measurement, temp_ori = self.get_measurement_full(model_phrase, new_ori)
            new_ori = temp_ori
            if measurement == 1:
                result['tp'] += 1
            else:
                result['fp'] += 1
            if not new_ori:
                break
        result['fn'] = len(new_ori)
        return result

    def evaluate(self, eval_type, entity_type=None):
        if entity_type is not None:
            print(entity_type)
        else:
            print()

        if eval_type == 'partial':
            print('PARTIAL EVALUATION')
            evaluation_function = self.partial_evaluator
        else:
            print("FULL EVALUATION")
            evaluation_function = self.full_evaluator

        original_result = None
        if self.task == 'ke':
            with open(self.DATA_PATH + 'original_keyphrases.json') as data_file:
                original_result = json.load(data_file)
        elif self.task == 'mer':
            with open(self.DATA_PATH + 'original_{}.json'.format(entity_type)) as data_file:
                original_result = json.load(data_file)

        fold_precision = []
        fold_recall = []
        fold_f_measure = []
        total_result = {'fp': 0.0, 'fn': 0.0, 'tp': 0.0}

        for fold_index in self.folds:
            model_result = {}
            result = {'fp': 0.0, 'fn': 0.0, 'tp': 0.0}

            file_path = self.path + 'fold_{}.json'.format(fold_index)
            with open(file_path) as data_file:
                model_result = json.load(data_file)

            for x, value in model_result.items():
                temp_res = evaluation_function(value, original_result[x])
                result['tp'] += temp_res['tp']
                result['fp'] += temp_res['fp']
                result['fn'] += temp_res['fn']

            precision = result['tp']/(result['tp'] + result['fp'])
            recall = result['tp']/(result['tp'] + result['fn'])
            if result['tp'] != 0:
                f_measure = 2 * (precision * recall) / (precision + recall)
            else:
                print('Fold {} has zero true positives'.format(fold_index))
                f_measure = 0

            fold_precision.append(precision)
            fold_recall.append(recall)
            fold_f_measure.append(f_measure)

            total_result['tp'] += result['tp']
            total_result['fp'] += result['fp']
            total_result['fn'] += result['fn']

            print('=== Fold {} ==='.format(fold_index))
            print('Precision: {:.5f}%'.format(precision))
            print('Recall: {:.5f}%'.format(recall))
            print('F-Measure: {:.5f}%'.format(f_measure))
            print()
        
        print('=== Average per Fold ===')
        print('Precision: {:.5f}%'.format(sum(fold_precision) / len(self.folds)))
        print('Recall: {:.5f}%'.format(sum(fold_recall) / len(self.folds)))
        print('F-Measure: {:.5f}%'.format(sum(fold_f_measure) / len(self.folds)))
        print()

        total_precision = total_result['tp']/(total_result['tp'] + total_result['fp'])
        total_recall = total_result['tp']/(total_result['tp'] + total_result['fn'])
        total_f_measure = 2 * (total_precision * total_recall) / (total_precision + total_recall)

        print('=== Overall ===')
        print('Precision: {:.5f}%'.format(total_precision))
        print('Recall: {:.5f}%'.format(total_recall))
        print('F-Measure: {:.5f}%'.format(total_f_measure))
        print()


class EvaluationTools:
    def __init__(self, path, multi_task=False):
        self.path = path
        self.multi_task = multi_task

        if not multi_task:
            self.DATA_PATH = DATA_PATH + 'single_task/'
        else:
            self.DATA_PATH = DATA_PATH + 'multi_task/'

    def print_evaluation(self, task, path, folds=[i for i in range(1, 11)], bio=True, other=False):
        golds = []
        predictions = []

        for fold_index in folds:
            with open(path + 'fold_{}.txt'.format(fold_index), 'r') as file_object:
                for line in file_object:
                    if not (line.startswith('Index:') or line == '\n'):
                        gold, prediction = line.split()[-2:]
                        golds.append(gold)
                        predictions.append(prediction)
        
        if task == 'sentence_recognition':
            labels = ['B-BACKGROUND', 'I-BACKGROUND', 'B-QUESTION', 'I-QUESTION', 'B-IGNORE', 'I-IGNORE']
        elif task == 'boundary_detection':
            labels = ['B', 'I']
        elif task == 'keyphrases_extraction':
            labels = ['BK', 'IK', 'OO']
        else:
            labels = ['B-Disease', 'B-Drug', 'B-Symptom', 'B-Treatment', 'I-Disease', 'I-Drug', 'I-Symptom', 'I-Treatment', 'OO']

        if not other and 'OO' in labels:
            labels.remove('OO')
        
        print(classification_report(golds, predictions, labels=labels, digits=4))
        print('\n{}'.format(labels))
        print(confusion_matrix(golds, predictions, labels=labels))
        return predictions

    def add_subword_count(self, length, gold, prediction):
        if length not in self.subword_true_count:
            self.subword_true_count[length] = [0, 0]

        self.subword_true_count[length][0] += 1
        if gold == prediction:
            self.subword_true_count[length][1] += 1
        
    def _post_v1(self, file_name, token_list):
        if len(token_list) == 0:
                return
        elif len(token_list) < 2:
            line = token_list[0]
            file_name.write('{} {} {}\n'.format(line[0], line[1], line[2]))
        else:
            pieces = [x[0] for x in token_list]
            golds = [x[1] for x in token_list]
            predictions = [x[2] for x in token_list]

            token = ''.join(pieces).replace('#', '')
            try:
                assert golds.count(golds[0]) == len(golds)
            except:
                print(golds)
            file_name.write('{} {} {}\n'.format(token, golds[0], predictions[0]))
            self.add_subword_count(len(token_list), golds[0], predictions[0])

    def _post_v2(self, file_name, token_list):
        if len(token_list) == 0:
                return
        elif len(token_list) < 2:
            line = token_list[0]
            file_name.write('{} {} {}\n'.format(line[0], line[1], line[2]))
        else:
            pieces = [x[0] for x in token_list]
            golds = [x[1] for x in token_list]
            predictions = [x[2] for x in token_list]

            token = ''.join(pieces).replace('#', '')
            try:
                assert golds.count(golds[0]) == len(golds)
            except:
                print(golds)
            file_name.write('{} {} {}\n'.format(token, golds[0], predictions[-1]))
            self.add_subword_count(len(token_list), golds[0], predictions[-1])

    def _post_v3(self, file_name, token_list):
        if len(token_list) == 0:
                return
        elif len(token_list) < 2:
            line = token_list[0]
            file_name.write('{} {} {}\n'.format(line[0], line[1], line[2]))
        else:
            pieces = [x[0] for x in token_list]
            golds = [x[1] for x in token_list]
            predictions = [x[2] for x in token_list]

            token = ''.join(pieces).replace('#', '')
            try:
                assert golds.count(golds[0]) == len(golds)
            except:
                print(golds)
            
            prediction_count = {}
            for prediction in predictions:
                prediction_count[prediction] = prediction_count.get(prediction, 0) + 1

            max_count = max(prediction_count.values())
            prediction_max_count = [key for key, value in prediction_count.items() if value == max_count]
            final_prediction = prediction_max_count[0]

            file_name.write('{} {} {}\n'.format(token, golds[0], final_prediction))
            self.add_subword_count(len(token_list), golds[0], final_prediction)

    def _post_v4(self, file_name, token_list):
        if len(token_list) == 0:
                return
        elif len(token_list) < 2:
            line = token_list[0]
            file_name.write('{} {} {}\n'.format(line[0], line[1], line[2]))
        else:
            pieces = [x[0] for x in token_list]
            golds = [x[1] for x in token_list]
            predictions = [x[2] for x in token_list]

            token = ''.join(pieces).replace('#', '')
            try:
                assert golds.count(golds[0]) == len(golds)
            except:
                print(golds)
            
            prediction_count = {}
            final_prediction = ''
            for prediction in predictions:
                if prediction.startswith('B'):
                    final_prediction = prediction
                    break
                else:
                    prediction_count[prediction] = prediction_count.get(prediction, 0) + 1

            if final_prediction is '':
                prediction_count.pop('OO', None)
                if len(prediction_count) < 1:
                    final_prediction = 'OO'
                else:
                    max_count = max(prediction_count.values())
                    prediction_max_count = [key for key, value in prediction_count.items() if value == max_count]
                    final_prediction = prediction_max_count[0]

            file_name.write('{} {} {}\n'.format(token, golds[0], final_prediction))
            self.add_subword_count(len(token_list), golds[0], final_prediction)

    def print_true_subword_prediction(self):
        for length in sorted(self.subword_true_count.keys()):
            value = self.subword_true_count[length]
            print('Length {}: {} correct out of {}'.format(length, value[1], value[0]))

    def _keyphrase_extractor(self, result):
        with open(self.DATA_PATH + 'words.json') as json_data:
            words_vector = json.load(json_data)

        result_key = {}
        for key, value in result.items():
            result_arr = value
            index_before = 0
            arr_key = []
            words = words_vector[key]

            keyphrase = ''
            for index in range(len(words)):
                if result_arr[index] != 0 and index_before == 0:
                    index_before = result_arr[index]
                    keyphrase += words[index] + ' '
                elif result_arr[index] == 0 and index_before != 0:
                    index_before = 0
                    arr_key.append(keyphrase)
                    keyphrase = ''
                elif result_arr[index] == 2 and index_before == 2:
                    index_before = 2
                    arr_key.append(keyphrase)
                    keyphrase = ''
                elif result_arr[index] == 1 and index_before != 0:
                    index_before = 1
                    keyphrase += words[index] + ' '
                elif result_arr[index] == 2 and index_before == 1:
                    index_before = 1
                    arr_key.append(keyphrase)
                    keyphrase = ''
                    keyphrase += words[index] + ' '
                if index == len(words) - 1 and index_before == 1:
                    arr_key.append(keyphrase)
            result_key[key] = arr_key
        return result_key

    def evaluate_keyphrases(self, path, folds=[i for i in range(1, 11)]):
        tag_dict = {'BK': 2, 'IK': 1, 'OO': 0}

        extract_folder = path + 'keyphrases/'
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        for fold_index in folds:
            predictions = {}
            index = None
            token_tags = []

            with open(path + 'fold_{}.txt'.format(fold_index), 'r') as file_object:
                for line in file_object:
                    if line.startswith('Index:'):
                        index = line.split(': ')[-1][:-1]
                    elif line == '\n' or line == '':
                        if not index:
                            continue
                        predictions[index] = token_tags
                        token_tags = []
                    else:
                        tag_predict = line.split()[-1]
                        token_tags.append(tag_dict[tag_predict])
                predictions[index] = token_tags

            keyphrases = self._keyphrase_extractor(predictions)
            with open(extract_folder + 'fold_{}.json'.format(fold_index), 'w') as outfile:
                json.dump(keyphrases, outfile, indent=2)

        evaluator = Evaluator(extract_folder, 'ke', folds, self.multi_task)
        evaluator.evaluate('partial')
        evaluator.evaluate('full')

    def evaluate_entity(self, path, entity_type, folds=[i for i in range(1, 11)]):
        extract_folder = path + '{}/'.format(entity_type)
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        for fold_index in folds:
            index = None
            predictions = {}
            entities = []
            current_entity = []

            with open(path + 'fold_{}.txt'.format(fold_index), 'r') as file_object:
                for line in file_object:
                    if line.startswith('Index:'):
                        index = line.split(': ')[-1][:-1]
                        continue
                    elif line == '\n' or line == '':
                        if not index:
                            continue
                        predictions[index] = entities
                        entities = []
                        continue

                    x = line.split()
                    token = ' '.join(x[:-2])
                    gold, prediction = x[-2:]
                    if prediction == 'I-{}'.format(entity_type):
                        current_entity.append(token)
                    elif prediction == 'B-{}'.format(entity_type):
                        if len(current_entity) > 0:
                            entities.append(' '.join(current_entity))
                        current_entity = [token]
                    else:
                        if len(current_entity) > 0:
                            entities.append(' '.join(current_entity))
                        current_entity = []
                if len(current_entity) > 0:
                    entities.append(' '.join(current_entity))        
                predictions[index] = entities
                current_entity = []
                entities = []

            with open(extract_folder + 'fold_{}.json'.format(fold_index), 'w') as outfile:
                json.dump(predictions, outfile, indent=2)

        evaluator = Evaluator(extract_folder, 'mer', folds, self.multi_task)
        evaluator.evaluate('partial', entity_type)
        evaluator.evaluate('full', entity_type)


    def post_evaluation(self, task, version=1, folds=[i for i in range(1, 11)]):
        if version == 1:
            post_function = self._post_v1
        elif version == 2:
            post_function = self._post_v2
        elif version == 3:
            post_function = self._post_v3
        elif version == 4:
            post_function = self._post_v4
        else:
            print('Post-processing version {} does not exist!'.format(version))
            return

        word_pieces = []
        self.subword_true_count = {}

        write_folder = self.path + 'post_v{}/'.format(version)
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)

        for fold_index in folds:
            file_read = self.path + 'fold_{}.txt'.format(fold_index)
            file_write = write_folder + 'fold_{}.txt'.format(fold_index)

            with open(file_read, 'r') as f_read, open(file_write, 'w') as f_write:
                for line in f_read:
                    if line.startswith('Index:'):
                        f_write.write('\n{}'.format(line))
                    elif line == '\n':
                        post_function(f_write, word_pieces)
                        f_write.write('')
                        word_pieces = []
                    elif line.startswith('##'):
                        x = line.split()
                        token = ' '.join(x[:-2])
                        gold, prediction = x[-2:]
                        word_pieces.append((token, gold, prediction))
                    else:
                        x = line.split()
                        token = ' '.join(x[:-2])
                        gold, prediction = x[-2:]
                        post_function(f_write, word_pieces)
                        word_pieces = [(token, gold, prediction)]

        predictions = self.print_evaluation(task, write_folder, folds)
        print()
        self.print_true_subword_prediction()
        if task == 'keyphrases_extraction':
            self.evaluate_keyphrases(write_folder, folds)
        elif task == 'medical_entity_recognition':
            for entity_type in ['Disease', 'Drug', 'Symptom', 'Treatment']:
                print()
                self.evaluate_entity(write_folder, entity_type, folds)
