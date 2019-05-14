"""
Wrapper class.
"""
from seqeval.metrics import f1_score

from anago.models import BiLSTMCRF, save_model, load_model
from anago.preprocessing import IndexTransformer
from anago.tagger import Tagger
from anago.trainer import Trainer
from anago.utils import filter_embeddings
import keras
from anago.genAravec import writeTupleArray,clean_str,get_vec,calc_vec,get_all_ngrams,get_ngrams,get_existed_tokens,checkerLen,vectorSim,AdjustPredTag,getAllPredTags
from gensim.models import KeyedVectors
import gensim
from seqeval.metrics import classification_report

import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
import anago.bertmodels as ABM

#from tensorflow.keras import backend as K

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
max_seq_length = 256

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )
    #print(do_lower_case)

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

class Sequence(object):

    def __init__(self,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True,
                 initial_vocab=None,
                 optimizer='adam',
                 layer2Flag=False,
                 layerdropout=0,
                 # fastArFlag=False,
                 # fastModelAr="",
                 # fastEnFlag=False,
                 # fastModelEn="",ArTwitterFlag=False,ArTwitterModel="",fileToWrite="Invalid.txt",
                 bretFlag=False,bretMaxLen=100,bert_path="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"):

        self.model = None
        self.p = None
        self.tagger = None

        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_lstm_size = word_lstm_size
        self.char_lstm_size = char_lstm_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.embeddings = embeddings
        self.use_char = use_char
        self.use_crf = use_crf
        self.initial_vocab = initial_vocab
        self.optimizer = optimizer
        self._layer2Flag = layer2Flag
        self._layerdropout = layerdropout
        # self._fastArFlag=fastArFlag
        # self._fastEnFlag=fastEnFlag
        # self._fastModelAr=fastModelAr
        # self._fastModelEn=fastModelEn
        # self._ArTwitterFlag=ArTwitterFlag
        # self._ArTwitterModel=ArTwitterModel
        # self._fileToWrite=fileToWrite
        self._bretFlag=bretFlag
        self._bretMaxLen=bretMaxLen
        self._bert_path=bert_path

    def bertFit(self,x_train, y_train,x_valid=None, y_valid=None,epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):

        sess = tf.Session()
        bert_path = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
        max_seq_length = self._bretMaxLen

        tokenizer = create_tokenizer_from_hub_module()
        print("tokenizar done")

        train_examples = convert_text_to_examples(x_train, y_train)

        (train_input_ids, train_input_masks, train_segment_ids, train_labels) = convert_examples_to_features(tokenizer,train_examples,max_seq_length=max_seq_length)

        model =ABM.BertBiLSTMCRF(
                          num_labels=9,
                          char_embedding_dim=self.char_embedding_dim,
                          word_lstm_size=self.word_lstm_size,
                          char_lstm_size=self.char_lstm_size,
                          fc_dim=self.fc_dim,
                          use_char=self.use_char,
                          char_vocab_size=None,
                          use_crf=self.use_crf,
                          layer2Flag=self._layer2Flag,
                          layerdropout=self._layerdropout,
                          bretFlag=self._bretFlag,
                          bretMaxLen=self._bretMaxLen,
                          bert_path=self._bert_path)



        model,loss = model.build()

        # Instantiate variables
        ABM.initialize_vars(sess)

        model.fit(
            [train_input_ids, train_input_masks, train_segment_ids],
            train_labels,
            epochs=epochs,
            batch_size=batch_size
        )

    def bertFitV2(self, x_train, y_train,x_valid=None, y_valid=None, epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):

        sess = tf.Session()
        bert_path = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
        max_seq_length = self._bretMaxLen

        p = IndexTransformer(initial_vocab=self.initial_vocab, use_char=self.use_char)
        p.fit(x_train, y_train)
        embeddings = filter_embeddings(self.embeddings, p._word_vocab.vocab, self.word_embedding_dim)

        #tokenizer = create_tokenizer_from_hub_module()
        #print("tokenizar done")

        #train_examples = convert_text_to_examples(x_train, y_train)

        #(train_input_ids, train_input_masks, train_segment_ids, train_labels) = convert_examples_to_features(tokenizer,train_examples,max_seq_length=max_seq_length)

        model = ABM.BertBiLSTMCRF(
            num_labels=p.label_size,
            char_embedding_dim=self.char_embedding_dim,
            word_lstm_size=self.word_lstm_size,
            char_lstm_size=self.char_lstm_size,
            fc_dim=self.fc_dim,
            use_char=self.use_char,
            char_vocab_size=None,
            use_crf=self.use_crf,
            layer2Flag=self._layer2Flag,
            layerdropout=self._layerdropout,
            bretFlag=self._bretFlag,
            bretMaxLen=self._bretMaxLen,
            bert_path=self._bert_path)

        model, loss = model.build()

        # Instantiate variables
        ABM.initialize_vars(sess)

        model.compile(loss=loss, optimizer=self.optimizer)

        trainer = Trainer(model, preprocessor=p)
        trainer.train(x_train, y_train, x_valid, y_valid,
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose, callbacks=callbacks,
                      shuffle=shuffle)

        self.p = p
        self.model = model

    def fit(self, x_train, y_train, x_valid=None, y_valid=None,
            epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        """Fit the model for a fixed number of epochs.

        Args:
            x_train: list of training data.
            y_train: list of training target (label) data.
            x_valid: list of validation data.
            y_valid: list of validation target (label) data.
            batch_size: Integer.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch). `shuffle` will default to True.
        """
        p = IndexTransformer(initial_vocab=self.initial_vocab, use_char=self.use_char)
        p.fit(x_train, y_train,bretFlag=self._bretFlag,max_len=self._bretMaxLen)
        embeddings = filter_embeddings(self.embeddings, p._word_vocab.vocab, self.word_embedding_dim)

        model = BiLSTMCRF(char_vocab_size=p.char_vocab_size,
                          word_vocab_size=p.word_vocab_size,
                          num_labels=p.label_size,
                          word_embedding_dim=self.word_embedding_dim,
                          char_embedding_dim=self.char_embedding_dim,
                          word_lstm_size=self.word_lstm_size,
                          char_lstm_size=self.char_lstm_size,
                          fc_dim=self.fc_dim,
                          dropout=self.dropout,
                          embeddings=embeddings,
                          use_char=self.use_char,
                          use_crf=self.use_crf,
                          layer2Flag=self._layer2Flag,
                          layerdropout=self._layerdropout,
                          bretFlag=self._bretFlag,
                          bretMaxLen=self._bretMaxLen,
                          bert_path=self._bert_path)
        model, loss = model.build()
        #if(self.optimizer.lower()=="adam"):
            #self.optimizer=keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1)
        model.compile(loss=loss, optimizer=self.optimizer)

        trainer = Trainer(model, preprocessor=p)
        trainer.train(x_train, y_train, x_valid, y_valid,
                      epochs=epochs, batch_size=batch_size,
                      verbose=verbose, callbacks=callbacks,
                      shuffle=shuffle)

        self.p = p
        self.model = model

    def score(self, x_test, y_test,fileToWrite):
        """Returns the f1-micro score on the given test data and labels.

        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.

            y_test : array-like, shape = (n_samples, sent_length)
            True labels for x.

        Returns:
            score : float, f1-micro score.
        """
        if self.model:
            # if(self._fastArFlag):
            #     ArText=KeyedVectors.load_word2vec_format(self._fastModelAr)
            # if(self._fastEnFlag):
            #     EnText=KeyedVectors.load_word2vec_format(self._fastModelEn)
            # if(self._ArTwitterFlag):
            #     ArTwitter=gensim.models.Word2Vec.load(self._ArTwitterModel)

            x_test_org=x_test
            x_test = self.p.transform(x_test)
            lengths = map(len, y_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            # adjust here
            # vector similarity approach

            # if(self._ArTwitterFlag and self._fastEnFlag):
            #     print("here")
            #     AdjustPredTag(t_model=ArTwitter,t_en_model=EnText,x_test_org=x_test_org,y_pred=y_pred,ratioSimilarity=0.6,topn=30)

            writeTupleArray(x_test_org,y_pred,fileToWrite)

            #checkerLen(x_test_org,y_pred)
            #print(y_pred)
            print(classification_report(y_test,y_pred))
            score = f1_score(y_test, y_pred)
            print("F-score is")
            return score
        else:
            raise OSError('Could not find a model. Call load(dir_path).')

    def analyze(self, text, tokenizer=str.split):
        """Analyze text and return pretty format.

        Args:
            text: string, the input text.
            tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.

        Returns:
            res: dict.
        """
        if not self.tagger:
            self.tagger = Tagger(self.model,
                                 preprocessor=self.p,
                                 tokenizer=tokenizer)

        return self.tagger.analyze(text)

    def save(self, weights_file, params_file, preprocessor_file):
        self.p.save(preprocessor_file)
        save_model(self.model, weights_file, params_file)

    @classmethod
    def load(cls, weights_file, params_file, preprocessor_file):
        self = cls()
        self.p = IndexTransformer.load(preprocessor_file)
        self.model = load_model(weights_file, params_file)

        return self
