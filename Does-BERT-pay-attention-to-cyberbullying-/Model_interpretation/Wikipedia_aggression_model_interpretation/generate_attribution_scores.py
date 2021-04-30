#!/usr/bin/env python
# coding: utf-8

# # Interpretation of BertForSequenceClassification in captum
# 
# In this notebook we use Captum to interpret a BERT sentiment classifier finetuned on the imdb dataset https://huggingface.co/lvwerra/bert-imdb 

# In[1]:


import captum


# In[2]:


from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


print('We will use the GPU:', torch.cuda.get_device_name(0))


# ## load data

# In[5]:


import pandas as pd

#wikipedia_aggression_train_df = pd.read_csv("../../Data/wikipedia_aggression/")
wikipedia_aggression_test_df = pd.read_csv("../../../Data/wikipedia_aggression/wp_agg_data_test_sample_1000.csv")

#wikipedia_aggression_train_df = wikipedia_aggression_train_df.dropna()
wikipedia_aggression_test_df = wikipedia_aggression_test_df.dropna()

print(len(wikipedia_aggression_test_df))

reduced_len_sentence = []
for i in wikipedia_aggression_test_df.Text_clean_for_BERT.values:
    s = i.split(" ")[:]
    reduced_len_sentence.append(" ".join(s))

wikipedia_aggression_test_df["reduced_text_clean"] = reduced_len_sentence

#data_test_pos = wikipedia_aggression_train_df[wikipedia_aggression_train_df["oh_label"] == 1]
#data_test_neg = wikipedia_aggression_test_df[wikipedia_aggression_test_df["oh_label"] == 0]

#wikipedia_aggression_test_df = wikipedia_aggression_test_df.sample(100)
#wikipedia_aggression_test_df.to_csv("../../Data/wikipedia_aggression/wikipedia_aggression_test_set_sample_100.csv", index=False)

# In[21]:


# load model
model = BertForSequenceClassification.from_pretrained('../../trained_models/BERT-Fine-Tuned/Pytorch/Fine_Tune_wtp_agg_clean_text128/')
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('../../trained_models/BERT-Fine-Tuned/Pytorch/Fine_Tune_wtp_agg_clean_text128/')


# In[22]:


def predict(inputs):
    #print('model(inputs): ', model(inputs))
    return model(inputs)[0]


# In[23]:


ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence


# In[24]:


def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):

    text_ids = tokenizer.encode(text, add_special_tokens=False, max_length=64,truncation=True)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


# In[25]:


def custom_forward(inputs):
    preds = predict(inputs)
    return torch.softmax(preds, dim = 1)[:, 1] # for negative attribution, torch.softmax(preds, dim = 1)[:, 1] <- for positive attribution


# In[26]:


lig = LayerIntegratedGradients(custom_forward, model.bert.embeddings)


# In[27]:


def get_attribution_for_test_set(lig, test_data_set):
    words_ls = []
    attributions_ls = []
    test_set_word_att_dict = {}
    
    for index, row in test_data_set.iterrows():
        text = row["Text"]
        clean_text = row["reduced_text_clean"]
        oh_label = row['oh_label']
        
        input_ids, ref_input_ids, sep_id = construct_input_ref_pair(clean_text, ref_token_id, sep_token_id, cls_token_id)
        token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
        position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
        attention_mask = construct_attention_mask(input_ids)

        indices = input_ids[0].detach().tolist()
        all_tokens = tokenizer.convert_ids_to_tokens(indices)
        print(input_ids.shape)
        attributions, delta = lig.attribute(inputs=input_ids,
                                    baselines=ref_input_ids,
                                    n_steps=5000,
                                    internal_batch_size=5,
                                    return_convergence_delta=True)
        tokenized_sen = tokenizer.tokenize(clean_text)
        print(len(tokenized_sen))
        tokenized_sen = tokenized_sen[:64]
        for i in tokenized_sen:
            word = i
            words_ls.append(word)
            index = tokenized_sen.index(i)+1
            attribution = float(sum(attributions[0][index]))
            attributions_ls.append(attribution)
            
    #words_ls_flatten = [item for sublist in words_ls for item in sublist]
    #attributions_ls_flatten = [item for sublist in attributions_ls for item in sublist]
    
    test_set_word_att_dict["words"] = words_ls
    test_set_word_att_dict["attribution"] = attributions_ls
    
    return test_set_word_att_dict


# In[28]:


#twitter_sample = pd.concat([data_test_pos.sample(50), data_test_neg.sample(50)])


# In[ ]:


test_set_word_att_dict = get_attribution_for_test_set(lig, wikipedia_aggression_test_df)


# In[ ]:


word_attribution_df = pd.DataFrame.from_dict(test_set_word_att_dict)


# In[ ]:


len(word_attribution_df)


# In[ ]:


word_attribution_df.head(10)


# In[ ]:


word_attribution_df["abs_attribution"] = [np.absolute(i) for i in word_attribution_df.attribution]

word_attribution_df.to_csv("attribution_scores/bert_with_fine_tuning/wikipedia_aggression_1000_data_fine_tuned_bert_attribution_scores.csv", index=False)


# In[ ]:



