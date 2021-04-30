import os
import pandas as pd
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as nnf

import torch
from transformers import *
from torch.utils.data import TensorDataset, random_split, SequentialSampler,DataLoader, RandomSampler
import pandas as pd
import numpy as np
from collections import defaultdict
# (logits, hidden_states, attentions)
import seaborn as sns; sns.set()
# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



def get_model_attention_for_sentence(sentences,single_sentence=False, add_special_token=False):
  # Tokenize all of the sentences and map the tokens to thier word IDs.
  attention_weights_per_sentence = []
  sentences_tokens = []
  if single_sentence == True:
    token = [tokenizer.cls_token] + tokenizer.tokenize(sentences)[:64] + [tokenizer.sep_token]
    inputids = torch.LongTensor([tokenizer.encode(token, add_special_tokens=add_special_token)])#, max_length=64,truncation=True,padding=true)])
    # Add the encoded sentence to the list.
    outputs = model(inputids)   # Forward pass, calculate logit predictions
    sentences_tokens.append(token)
    attention_weights_per_sentence.append(outputs[2])
  else:
    # For every sentence...
    for sent in sentences:
        token = [tokenizer.cls_token] + tokenizer.tokenize(sent)[:64] + [tokenizer.sep_token]
        inputids = torch.LongTensor([tokenizer.encode(token, add_special_tokens=add_special_token)])#, max_length=64,truncation=True,padding=True)])
        # Add the encoded sentence to the list.
        outputs = model(inputids)   # Forward pass, calculate logit predictions
        sentences_tokens.append(token)
        attention_weights_per_sentence.append(outputs[2])
  return sentences_tokens, attention_weights_per_sentence


def get_attention_map_per_word_as_df (sentences_tokenized, attention_list_per_sentence):
  #get attention map for words in sentences
  attention_map_per_word = defaultdict(list)
  for token_sentence in sentences_tokenized:
    sentence_index = sentences_tokenized.index(token_sentence)
    print(len(token_sentence))
    for token in token_sentence:
      word_index = token_sentence.index(token)
      #print(token)
      for layer in range(0,12):
        for head in range(0,12):
          attention_weight_per_layer_per_head_per_word = attention_list_per_sentence[sentence_index][layer][0][head]
          normalized_attention_weight_per_layer_per_head_per_word = torch.sum(attention_weight_per_layer_per_head_per_word,dim=0)/attention_weight_per_layer_per_head_per_word.shape[0]
          #print(len(normalized_attention_weight_per_layer_per_head_per_word))
          attention_map_per_word["word"].append(token)
          attention_map_per_word["attention_map"].append({layer+1:{head+1:float(normalized_attention_weight_per_layer_per_head_per_word[word_index])}})
  attention_map_per_word_df = pd.DataFrame.from_dict(attention_map_per_word, orient='index')
  attention_map_per_word_df = attention_map_per_word_df.transpose()
  return attention_map_per_word_df

def generate_attention_attention_layers_per_head_for_word(word,attention_map_df):
  import ast
  frames = []
  #len(attention_map_df[attention_map_df["word"]==word]["attention_map"].values)
  for i in attention_map_df[attention_map_df["word"]==word]["attention_map"].values:
    frames.append(pd.DataFrame.from_dict(i, orient='index'))
  res = pd.concat(frames)
  res = res.fillna(0)
  res = res.groupby(res.index).mean()
  return res

import torch
#output_dir = 'bert-base-uncased'
output_dir = "../../trained_models/BERT-Fine-Tuned/Pytorch/Fine_Tune_Kaggle_clean_text128_no_stem/"

config = BertConfig.from_pretrained(output_dir, output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)

model = BertForSequenceClassification.from_pretrained(output_dir, config=config)
# Load the dataset into a pandas dataframe.
data_test = pd.read_csv("../../../Data/Kaggle/kaggle_data_test_sample_1000.csv")
data_test = data_test.dropna()

reduced_len_sentence = []
for i in data_test.Text_clean_for_BERT.values:
    s = i.split(" ")[:]
    reduced_len_sentence.append(" ".join(s))

data_test["reduced_text_clean"] = reduced_len_sentence

sentences_lst_tn = data_test.reduced_text_clean.values
sentences_tokenized_lst_tn, attention_list_per_sentence_lst_tn = get_model_attention_for_sentence(sentences_lst_tn)
attention_map_per_word_lst_tn_df = get_attention_map_per_word_as_df(sentences_tokenized_lst_tn, attention_list_per_sentence_lst_tn )

attention_map_per_word_lst_tn_df.to_csv("attention_maps/bert_with_fine_tuning/attention_weights_for_1000_data.csv",index=False)