# Tasks
1. Sentence Recognition
    - [One-Step](https://github.com/tsqfnfl/tugas-akhir/blob/12d353d068459a47794f4ee46563cb9c02ae4743/tasks/single_tasks.py#L3-L9), receives sequence of tokens as input and label each token as **B-BACKGROUND**, **I-BACKGROUND**, **B-QUESTION**, **I-QUESTION**, **B-IGNORE**, or **I-IGNORE**.
    - Two-Step
        - [Boundary Detection](https://github.com/tsqfnfl/tugas-akhir/blob/12d353d068459a47794f4ee46563cb9c02ae4743/tasks/single_tasks.py#L11-L17), receives sequence of tokens as input and label each token as **B** (the beginning of a sentence) or **I** (not the beginning of a sentence).
        - [Sentence Classification](https://github.com/tsqfnfl/tugas-akhir/blob/master/tasks/sentence_classification.py), receives sequence of sentences as input and label each sentence as **BACKGROUND**, **QUESTION**, or **IGNORE**.
2. [Medical Entity Recognition](https://github.com/tsqfnfl/tugas-akhir/blob/12d353d068459a47794f4ee46563cb9c02ae4743/tasks/single_tasks.py#L19-L25), receives sequence of tokens as input and label each token as **B-DISEASE**, **I-DISEASE**, **B-DRUG**, **I-DRUG**, **B-SYMPTOM**, **I-SYMPTOM**, **B-TREATMENT**, **I-TREATMENT**, or **OO**.
3. [Keyphrases Extraction](https://github.com/tsqfnfl/tugas-akhir/blob/12d353d068459a47794f4ee46563cb9c02ae4743/tasks/single_tasks.py#L28-L34), receives sequence of tokens as input and label each token as **BK**, **IK**, or **OO**.
4. [Multi-Task Learning](https://github.com/tsqfnfl/tugas-akhir/blob/master/tasks/multi_task.py) for Medical Entity Recognition and Keyphrases Extraction.


# BERT
1. [IndoDistilBERT](https://huggingface.co/cahya/distilbert-base-indonesian), set `distil_bert = True` when initialize task object to use it.
2. [IndoBERT by IndoLEM](https://huggingface.co/indolem/indobert-base-uncased), set `distil_bert = False` and `indonlu = False` when initialize task object to use it.
3. [IndoBERT by IndoNLU (Base Version)](https://huggingface.co/indobenchmark/indobert-base-p1), set `distil_bert = False` and `indonlu = True`, and `base_version = True` when initialize task object to use it.
4. [IndoBERT by IndoNLU (Large Version)](https://huggingface.co/indobenchmark/indobert-large-p1), set `distil_bert = False` and `indonlu = True`, and `base_version = False` when initialize task object to use it.


# Models
There are three models that can be used for one-step sentence recognition, boundary detection, medical entity recognition, and keyphrases extraction:
1. [BERT](https://github.com/tsqfnfl/tugas-akhir/blob/12d353d068459a47794f4ee46563cb9c02ae4743/model_arch.py#L35-L53)
2. [BERT-BiLSTMs](https://github.com/tsqfnfl/tugas-akhir/blob/12d353d068459a47794f4ee46563cb9c02ae4743/model_arch.py#L35-L53)
3. [BERT-BiLSTMs-CRF](https://github.com/tsqfnfl/tugas-akhir/blob/12d353d068459a47794f4ee46563cb9c02ae4743/model_arch.py#L76-L90)

For multi-task learning, you can use regular [BERT-BiLSTMs](https://github.com/tsqfnfl/tugas-akhir/blob/12d353d068459a47794f4ee46563cb9c02ae4743/model_arch.py#L101-L127) or use [medical entity as keyphrases extraction's feature](https://github.com/tsqfnfl/tugas-akhir/blob/12d353d068459a47794f4ee46563cb9c02ae4743/model_arch.py#L129-L157)


# Post-Processing
When a token is split into word-pieces during the tokenization process, there are four rules that can be used to decide the final label:
1. Set label prediction for the first word-piece as final label
2. Set label prediction for the last word-piece as final label
3. Set majority label prediction among word-pieces as final label. If two or more labels have same count, choose the label that appears first.
4. Use following rules to decide the final label:
    - If there is one or more B-<...> label, choose B-<...> that appears first as final label.
    - If there is no B-<...> label, choose I-<...> with the most occurences as final label.
    - If there is more than one I-<...> label with the most occurences, choose I-<...> label that appears first as final label.
    - If there is no B-<...> or I-<...>, label, choose OO as final label.

To use one of those rules, pass corresponding rule number (1, 2, 3, or 4) as argument for _version_ parameter when calling task object's `post_evaluate` function.

# Notes
For a more detailed guide, take a look at [example.ipynb](https://github.com/tsqfnfl/tugas-akhir/blob/master/example.ipynb) file.
