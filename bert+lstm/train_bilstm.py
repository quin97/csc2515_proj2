#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from transformers import AdamW, get_linear_schedule_with_warmup,BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# In[2]:


# get_ipython().system('nvidia-smi')


# In[3]:


train_df = pd.read_csv("train.csv", index_col=0, nrows=100000)
train_df = train_df.replace(np.nan, '', regex=True)
train_df.shape
train_df.head()
# train_df = train_df.dropna(subset=['title', 'abstract', 'general_category'])
train_df["trainingText"] = train_df['title']+train_df['abstract']+train_df['author_clean']

y = train_df['general_category']
X = train_df.drop('general_category', axis=1, inplace=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=5)
allcats = sorted(list(set(train_df['general_category'])))

val_set = pd.concat([X_val,y_val],axis = 1)
val_set.to_csv("validation_lstm.csv",sep="\t")
# columns_list = [1] # index numbers of columns you want to delete
# y_train = y_train.tolist()
# y_train = y_val.tolist()
# print ("y_train", y_train)
# print ("y_val", y_val)
y_train_ind = y_train.apply(lambda x: allcats.index(x))
y_val_ind = y_val.apply(lambda x: allcats.index(x))



train_labels = torch.tensor(y_train_ind.tolist())
val_labels = torch.tensor(y_val_ind.tolist())

print ("train_labels",train_labels)
print ("val_labels", val_labels)


# y_val_ind
# allcats
# y_val_ind[1]
unique_labels_val, unique_counts = np.unique(y_val_ind, return_counts=True)
print ("unique_labels",unique_labels_val)
print ("unique_counts", unique_counts)
print ("total unique labels", len(unique_counts))

unique_labels_train, unique_counts = np.unique(y_train_ind, return_counts=True)
print ("unique_labels",unique_labels_train)
print ("unique_counts", unique_counts)
print ("total unique labels", len(unique_counts))

# all_labels = unique_labels_train + unique_labels_val
all_labels = np.concatenate((unique_labels_train, unique_labels_val), axis=0)
print ("all_labels", np.unique(all_labels))
print ("labels_len", len(np.unique(all_labels)))
labels_len = len(np.unique(all_labels))


# In[4]:


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


# In[7]:


# allX = np.concatenate([X_train.trainingText.values, X_val.trainingText.values])
# encodedX = [tokenizer.encode(sent, add_special_tokens=True) for sent in allX]
# max_len = max([len(sent) for sent in encodedX])
# print('Max length: ', max_len)


# In[8]:


MAX_LEN = 256
device = "cuda:0"
# device = "cpu"


# In[9]:


train_inputs, train_masks = preprocessing_for_scibert(X_train['trainingText'])
val_inputs, val_masks = preprocessing_for_scibert(X_val['trainingText'])



# In[12]:


batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# In[13]:


# print ("max_label", max_label)

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
        D_in, H, D_out = 768, 50, labels_len

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=labels_len,output_attentions=False,
                                              output_hidden_states=False)

        self.lstm = nn.LSTM(D_in, H, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(H*2, D_out)

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

model = BertClassifier()
model.to(device)


# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# In[15]:


optimizer = AdamW(model.parameters(),lr=5e-5,eps=1e-8)
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


# In[16]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[17]:


seed_val = 5
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
loss_fn = nn.CrossEntropyLoss()

training_stats = []

total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0_epoch, t0_batch = time.time(), time.time()


    total_loss, batch_loss, batch_counts = 0, 0, 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        batch_counts +=1
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()


        logits = model(b_input_ids, attention_mask=b_input_mask)
        loss = loss_fn(logits, b_labels)
        batch_loss += loss.item()
        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()
        
        if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            time_elapsed = time.time() - t0_batch
            print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
            batch_loss, batch_counts = 0, 0
            t0_batch = time.time()
                
                
    avg_train_loss = total_loss / len(train_dataloader)

    training_time = format_time(time.time() - total_t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))


    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in val_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_input_mask)
        loss = loss_fn(logits, b_labels)
        
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(val_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

torch.save(model, 'model_arxiv_bilstm.pt')
print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats.to_csv("training_stats_bilstm.csv")
sns.set(font_scale=1.5)
sns.set_style("ticks")
plt.rcParams["figure.figsize"] = (12,6)

plt.plot(df_stats['Training Loss'], 'o-',color="royalblue", label="Training Loss")
plt.plot(df_stats['Valid. Loss'], 'o-', color="lightcoral", label="Validation Loss")

plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.savefig("scibert+bilstm_epoch8_batch8.png")

