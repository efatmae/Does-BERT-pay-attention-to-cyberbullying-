from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from pretrained_models_training import *
import pandas as pd

# read data
#train_df = pd.read_csv("../../../Data/Kaggle/kaggle_data_train.csv", index_col=False)
test_df = pd.read_csv("../../../Data/Kaggle/kaggle_data_test.csv", index_col=False)

#train_df = train_df.drop_duplicates(subset="Text_clean_for_BERT")
#test_df = test_df.drop_duplicates(subset="Text_clean_for_BERT")

#train_df = train_df.dropna()
#test_df = test_df.dropna()
maxlen = 128
batch_size = 32
no_iterations = 5
saver_name = "BERT_NFT_Kaggle"
results_files = "results/BERT_NFT/kaggle_dataset_no_stem.txt"

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
    output_hidden_states = True # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

predicted_dataset = Test_model_NFT(model,test_df,"Text_clean_for_BERT","oh_label",tokenizer, maxlen,batch_size,
                      saver_name, results_files)

predicted_dataset.to_csv("../../../Data/Kaggle/kaggle_data_test.csv",index=False)