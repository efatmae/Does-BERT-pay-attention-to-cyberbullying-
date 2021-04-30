import gensim
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel

def get_UD_embeddings(filename, word_dictionary, vocab_size, embedding_dimension):
    UD_model = gensim.models.KeyedVectors.load_word2vec_format(filename)
    embedding_matrix = np.zeros((vocab_size, embedding_dimension))
    for word, i in word_dictionary.items():
        if word in UD_model.wv.vocab:
            embedding_vector = UD_model.wv.get_vector(word=word)
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_chan_embeddings(filename, word_dictionary, vocab_size, embedding_dimension):
    chan_model = gensim.models.Word2Vec.load(filename)
    chan_embedding_matrix = np.zeros((vocab_size, embedding_dimension))
    for word, i in word_dictionary.items():
        if word in chan_model.wv.vocab:
            embedding_vector = chan_model.wv.get_vector(word=word)
            # words not found in embedding index will be all-zeros.
            chan_embedding_matrix[i] = embedding_vector
    return chan_embedding_matrix

def get_Glove_embeddings(filename, word_dictionary, embedding_dimension):
    embeddings_index = {}
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    glove_embedding_matrix = np.zeros((len(word_dictionary) + 1, embedding_dimension))
    for word, i in word_dictionary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            glove_embedding_matrix[i] = embedding_vector

    return glove_embedding_matrix

def get_sswe_embeddings(filename, word_dictionary, vocab_size, embedding_dimension):
    sswe_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    sswe_embedding_matrix = np.zeros((vocab_size, embedding_dimension))
    for word, i in word_dictionary.items():
        if word in sswe_model.wv.vocab:
            embedding_vector = sswe_model.wv.get_vector(word=word)
            # words not found in embedding index will be all-zeros.
            sswe_embedding_matrix[i] = embedding_vector
    return sswe_embedding_matrix

def get_google_news_embeddings(filename, word_dictionary, vocab_size, embedding_dimension):
    sswe_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    sswe_embedding_matrix = np.zeros((vocab_size, embedding_dimension))
    for word, i in word_dictionary.items():
        if word in sswe_model.wv.vocab:
            embedding_vector = sswe_model.wv.get_vector(word=word)
            # words not found in embedding index will be all-zeros.
            sswe_embedding_matrix[i] = embedding_vector
    return sswe_embedding_matrix

def get_bert_concat_emb_last_4_layers(model, tokenizer, text_df, maxlen, add_special_tokens):
    model.eval()
    tokenized = [tokenizer.encode(x, add_special_tokens=add_special_tokens, max_length=maxlen, truncation=True) for x in list(text_df)]
    padded = np.array([i + [0] * (64 - len(i)) for i in tokenized])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        model_output = model(input_ids, attention_mask=attention_mask)

    hidden_states = model_output.hidden_states
    features_last_1 = hidden_states[-1][:, :, :].numpy()
    features_last_2 = hidden_states[-2][:, :, :].numpy()
    features_last_3 = hidden_states[-3][:, :, :].numpy()
    features_last_4 = hidden_states[-4][:, :, :].numpy()
    features = np.concatenate((features_last_1, features_last_2, features_last_3, features_last_4), axis=2)
    return features