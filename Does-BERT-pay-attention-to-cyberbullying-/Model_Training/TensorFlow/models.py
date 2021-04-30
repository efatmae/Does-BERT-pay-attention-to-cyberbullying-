import tensorflow as tf
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional #,Merge
from keras.models import Model,Sequential

from keras import backend as K
from keras.engine.topology import Layer
from keras.regularizers import L1L2
from transformers import *

def LR(inp_dim):
    print("Model LR")
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=inp_dim
                    , kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                    , bias_regularizer=L1L2(l2=0.00000001)))
    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #print(model.summary())
    return model

def MLP(inp_dim, vocab_size, embed_size,use_word_embeddings=False,embedding_matrix=None, embedding_trainable=False):
#     K.clear_session()
    model = Sequential()
    if use_word_embeddings == True:
        model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=inp_dim, trainable=embedding_trainable))
        model.add(Flatten())
        model.add(Dropout(0.50))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
        model.add(Flatten())
        model.add(Dense(128, activation = 'relu', kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                                                , bias_regularizer=L1L2(l2=0.00000001)))
        model.add(Dense(64, activation='relu' , kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                        , bias_regularizer=L1L2(l2=0.00000001)))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                        , bias_regularizer=L1L2(l2=0.00000001)))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    #print(model.summary())
    return model

def lstm_keras(inp_dim, vocab_size, embed_size,use_word_embeddings=False,embedding_matrix=None, embedding_trainable=False):
#     K.clear_session()
    model = Sequential()
    if use_word_embeddings == True:
        model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=inp_dim, trainable=embedding_trainable))
    else:
        model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(LSTM(embed_size, kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                                                , bias_regularizer=L1L2(l2=0.00000001)))
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    #print (model.summary())
    return model

def lstm_bert (embed_size):
    model = Sequential()
    model.add(LSTM(embed_size, kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                                                , bias_regularizer=L1L2(l2=0.00000001)))
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model
### CNN #############
#  def cnn(inp_dim, vocab_size, embed_size, num_classes, learn_rate):
#      tf.reset_default_graph()
#      network = input_data(shape=[None, inp_dim], name='input')
#      network = tflearn.embedding(network, input_dim=vocab_size, output_dim=embed_size, name="EmbeddingLayer")
#      network = dropout(network, 0.25)
#      branch1 = conv_1d(network, embed_size, 3, padding='valid', activation='relu', regularizer="L2", name="layer_1")
#      branch2 = conv_1d(network, embed_size, 4, padding='valid', activation='relu', regularizer="L2", name="layer_2")
#      branch3 = conv_1d(network, embed_size, 5, padding='valid', activation='relu', regularizer="L2", name="layer_3")
#      network = merge([branch1, branch2, branch3], mode='concat', axis=1)
#      network = tf.expand_dims(network, 2)
#      network = global_max_pool(network)
#      network = dropout(network, 0.50)
#      network = fully_connected(network, num_classes, activation='softmax', name="fc")
#      network = regression(network, optimizer='adam', learning_rate=learn_rate,
#                           loss='categorical_crossentropy', name='target')
# #
#      model = tflearn.DNN(network, tensorboard_verbose=0)
#      return model

def blstm(inp_dim,vocab_size, embed_size,use_word_embeddings=False,embedding_matrix=None, embedding_trainable=False):
#     K.clear_session()
    model = Sequential()
    if use_word_embeddings == True:
        print("use word embedding is True")
        model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=inp_dim, trainable=embedding_trainable))
    else:
        model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size, kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                                                , bias_regularizer=L1L2(l2=0.00000001))))
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model

def bilstm_bert(embed_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(embed_size, kernel_regularizer=L1L2(l1=0.0, l2=0.00000001)
                                 , bias_regularizer=L1L2(l2=0.00000001))))
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
class AttLayer(Layer):

    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1],),
                                      initializer='random_normal',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def blstm_atten(inp_dim, vocab_size, embed_size,use_word_embeddings=False,embedding_matrix=None, embedding_trainable=False):
#     K.clear_session()
    model = Sequential()
    if use_word_embeddings == True:
        model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=inp_dim, trainable=embedding_trainable))
    else:
        model.add(Embedding(vocab_size, embed_size, input_length=inp_dim))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size, return_sequences=True)))
    model.add(AttLayer())
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))
    #adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    #model.summary()
    return model

def bert_fine_tuned(learn_rate=3e-5, epsilon=1e-08, clipnorm=1.0):
    bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")  # Automatically loads the config
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, epsilon=epsilon, clipnorm=clipnorm)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    bert_model.summary()
    return bert_model

def bert_without_fine_tuning():
    bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")
    #bert_model.summary()
    return bert_model

def distilbert_fine_tuning(learn_rate=3e-5, epsilon=1e-08, clipnorm=1.0):
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, epsilon=epsilon, clipnorm=clipnorm)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    #model.summary()
    return model