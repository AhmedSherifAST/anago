"""
Model definition.
"""
import json
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, TimeDistributed
from keras.layers.merge import Concatenate
from keras.models import Model, model_from_json
from bert_embedding import BertEmbedding


from anago.layers import CRF
import tensorflow  as tf
import tensorflow_hub as hub





def save_model(model, weights_file, params_file):
    with open(params_file, 'w') as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)
        model.save_weights(weights_file)


def load_model(weights_file, params_file):
    with open(params_file) as f:
        model = model_from_json(f.read(), custom_objects={'CRF': CRF})
        model.load_weights(weights_file)

    return model



class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10,bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self._bert_path=bert_path
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self._bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [tf.keras.backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)




class BertBiLSTMCRF(object):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self,
                 num_labels,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 char_vocab_size=None,
                 fc_dim=100,
                 use_char=False,
                 use_crf=True,
                 layer2Flag=False,
                 layerdropout=0,bretFlag=False,bretMaxLen=100,
                 bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):
        """Build a Bi-LSTM CRF model.

        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): character LSTM feature extractor output dimensions.
            char_lstm_size (int): word tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
            use_crf (boolean): use crf as last layer.
        """
        super(BertBiLSTMCRF).__init__()
        self._char_embedding_dim = char_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._fc_dim = fc_dim
        self._use_char = use_char
        self._use_crf = use_crf
        self._num_labels = num_labels
        self._layer2Flag=layer2Flag
        self._layerdropout=layerdropout
        self._bretFlag=bretFlag
        self._bretMaxLen=bretMaxLen
        self._bert_path=bert_path

    def build(self):
        # build word embedding


        #in_id = Input(shape=(self._bretMaxLen,), name="input_ids")
        #in_mask = Input(shape=(self._bretMaxLen,), name="input_masks")
        #in_segment = Input(shape=(self._bretMaxLen,), name="segment_ids")
        #inputs = [in_id, in_mask, in_segment]


        #word_embeddings = BertLayer(n_fine_tune_layers=3,bert_path=self._bert_path)(inputs)
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        word_embeddings = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')(word_ids)

        # build character based word embedding
        # if self._use_char:
        #     print("char Embedding layer On")
        #     char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
        #     inputs.append(char_ids)
        #     char_embeddings = Embedding(input_dim=self._char_vocab_size,
        #                                 output_dim=self._char_embedding_dim,
        #                                 mask_zero=True,
        #                                 name='char_embedding')(char_ids)
        #     char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))(char_embeddings)
        #     word_embeddings = Concatenate()([word_embeddings, char_embeddings])
        #
        #     word_embeddings = Dropout(self._dropout)(word_embeddings)


        z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True,dropout=self._layerdropout, recurrent_dropout=self._layerdropout))(word_embeddings)
        if(self._layer2Flag):
            z=Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True,dropout=self._layerdropout, recurrent_dropout=self._layerdropout))(z)
        z = Dense(self._fc_dim, activation='tanh')(z)

        if self._use_crf:
            crf = CRF(self._num_labels, sparse_target=False)
            loss = crf.loss_function
            pred = crf(z)
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)

        model = Model(inputs=inputs, outputs=pred)

        return model, loss




