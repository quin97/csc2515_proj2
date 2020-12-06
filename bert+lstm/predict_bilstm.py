import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import time
import datetime
import re
import random
import sklearn

from typing import Dict, Optional, List, Any
# !pip install transformers
# !pip install allennlp
from transformers import *
from allennlp.data import Vocabulary
from allennlp.models.model import Model
# from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder
# from allennlp.nn import InitializerApplicator, RegularizerApplicator
# from allennlp.nn import util
# from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, WordpieceTokenizer
from transformers import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM,BertForSequenceClassification

MAX_LEN = 256
device = "cuda:0"
# train_df = pd.read_csv("train.csv", index_col=0, nrows = 100000)
# train_df = train_df.replace(np.nan, '', regex=True)
# train_df.shape
# train_df.head()
# # train_df = train_df.dropna(subset=['title', 'abstract', 'general_category'])
# train_df["trainingText"] = train_df['title']+train_df['abstract']+train_df['author_clean']
#
# y = train_df['general_category']
# X = train_df.drop('general_category', axis=1, inplace=False)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=5)
#


class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 17

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=17,output_attentions=False,
                                              output_hidden_states=False)

        self.lstm = nn.LSTM(D_in, H, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(H*2, D_out)

        # Instantiate an one-layer feed-forward classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(D_in, H),
#             nn.ReLU(),
#             # nn.Dropout(0.5),
#             nn.Linear(H, D_out)
#         )

        # # Freeze the BERT model
        # if freeze_bert:
        #     for param in self.bert.parameters():
        #         param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
#         print ("last_hidden_state_cls", last_hidden_state_cls.shape[0])
        lstm_out, _ = self.lstm(last_hidden_state_cls.view(-1,last_hidden_state_cls.shape[0],768))
#         print ("lstm out", lstm_out.shape)
        logits = self.fc(lstm_out)
        # Feed input to classifier to compute logits
#         logits = self.classifier(last_hidden_state_cls)
#         print ("logits", logits[0].shape)
        return logits[0]

# model = BertClassifier()
# model.to(device)


def get_clean(x):
	# Remove all characters not in the English alphabet
	x = re.sub(r"[^a-zA-Z0-9]+", ' ', x)
	x = str(x).lower()
	x = re.sub(r'\s+', ' ',x).strip()
	return x


# In[5]:


tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)


# In[6]:


def preprocessing_for_scibert(data):

	input_ids = []
	attention_masks = []

	for sent in data:
		encoded_sent = tokenizer.encode_plus(text=get_clean(sent),add_special_tokens=True,
											 max_length=MAX_LEN, pad_to_max_length = True,return_attention_mask=True)

		input_ids.append(encoded_sent.get('input_ids'))
		attention_masks.append(encoded_sent.get('attention_mask'))

	input_ids = torch.tensor(input_ids)
	attention_masks = torch.tensor(attention_masks)
#     labels = torch.linspace(0, 4, steps=5)
	return input_ids, attention_masks


def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))

	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))


# In[17]:

def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	print ("pred_flat", pred_flat)
	labels_flat = labels.flatten()
	print ("labels_flat", labels_flat)
	return np.sum(pred_flat == labels_flat) / len(labels_flat)



test_df = pd.read_csv("./test.csv", index_col=0, nrows=10000)
test_df = test_df.replace(np.nan, '', regex=True)
test_df.shape
test_df.head()
# train_df = train_df.dropna(subset=['title', 'abstract', 'general_category'])
test_df["trainingText"] = test_df['title']+test_df['abstract']+test_df['author_clean']

print ("test_df]", test_df["trainingText"])
y_test = test_df['general_category']
X_test = test_df.drop('general_category', axis=1, inplace=False)
allcats = sorted(list(set(test_df['general_category'])))
y_test_ind = y_test.apply(lambda x: allcats.index(x))
test_labels = torch.tensor(y_test_ind.tolist())

test_inputs, test_masks = preprocessing_for_scibert(X_test['trainingText'])

batch_size = 8

prediction_data = TensorDataset(test_inputs, test_masks,test_labels)
prediction_dataloader = DataLoader(prediction_data, sampler=SequentialSampler(prediction_data), batch_size=batch_size)

model = torch.load('model_arxiv_bilstm.pt')
model.to(device)
model.eval()
print (model)

# Tracking variables
predictions , true_labels = [], []
total_eval_accuracy = 0

for batch in prediction_dataloader:
	# batch = tuple(t.to(device) for t in batch)
	# b_input_ids, b_input_mask , b_labels = batch
	b_input_ids = batch[0].to(device)
	b_input_mask = batch[1].to(device)
	b_labels = batch[2].to(device)

	with torch.no_grad():
		logits = model(b_input_ids,attention_mask=b_input_mask)

	logits = logits.detach().cpu().numpy()
	label_ids = b_labels.to('cpu').numpy()
	predictions.append(logits)
	true_labels.append(label_ids)

	total_eval_accuracy += flat_accuracy(logits, label_ids)
	print ("total_eval_accuracy", total_eval_accuracy)

from sklearn.metrics import matthews_corrcoef

matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# For each input batch...
for i in range(len(true_labels)):
    pred_labels_i = np.argmax(predictions[i],axis = 1).flatten()
    matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
    matthews_set.append(matthews)

matthews_df = pd.DataFrame(matthews_set)
matthews_df.to_csv("matthews_bilstm.csv")

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions,axis = 1).flatten()
flat_true_labels = np.concatenate(true_labels, axis=0)

dfpred = pd.DataFrame(flat_predictions)
dfpred.to_csv("predictions_bilstm.csv")
cr = sklearn.metrics.classification_report(flat_true_labels, flat_predictions)
cm = np.array2string(sklearn.metrics.confusion_matrix(flat_true_labels, flat_predictions))
f = open('report_bilstm.txt', 'w')
f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
f.close()

from sklearn.metrics import accuracy_score, hamming_loss
print('Hamming loss : {}'.format(hamming_loss(flat_true_labels, flat_predictions)))
print("Accuracy_score : {}".format(accuracy_score(flat_true_labels, flat_predictions)))
                
#print ("prediction_dataloader", len(prediction_dataloader))
#avg_val_accuracy = total_eval_accuracy / len(prediction_dataloader)
#print("  Accuracy: {0:.2f}".format(avg_val_accuracy))


