from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from pretrained_models_training import *
import pandas as pd

# read data
train_df = pd.read_csv("../../../Data/wikipedia_toxicity/wp_tox_data_train.csv", index_col=False)
test_df = pd.read_csv("../../../Data/wikipedia_toxicity/wp_tox_data_test.csv", index_col=False)

train_df = train_df.drop_duplicates(subset="Text_clean")
test_df = test_df.drop_duplicates(subset="Text_clean")

train_df = train_df.dropna()
test_df = test_df.dropna()

batch_size = 32
no_epochs = 10
no_iterations = 5
maxlen = 128
lr = 2e-5
eps = 1e-8
saver_name = "../../trained_models/BERT-Fine-Tuned/Pytorch/Fine_Tune_wtp_tox_clean_text128"
results_files = "results/BERT_Fine_Tuned/wtp_tox_dataset.txt"

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('../../trained_models/bert-base-uncased', do_lower_case=True)

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "../../trained_models/bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = True, # Whether the model returns attentions weights.
    output_hidden_states = True, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

predicted_dataset = train_model(model,train_df,test_df,"Text_clean_for_BERT","oh_label",tokenizer, maxlen, 0.3,
                batch_size, no_epochs, no_iterations, lr, eps,
                      saver_name, results_files)

predicted_dataset.to_csv("../../../Data/wikipedia_toxicity/wp_tox_data_test.csv",index=False)