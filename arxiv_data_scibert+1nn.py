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
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



# In[2]:


# get_ipython().system('nvidia-smi')


# In[3]:


train_df = pd.read_csv("train.csv", index_col=0)
# train_df = train_df.dropna(subset=['title', 'abstract', 'general_category'])
train_df.shape
train_df.head()
train_df = train_df.dropna(subset=['title', 'abstract', 'general_category'])
train_df["trainingText"] = train_df['title']+train_df['abstract']+train_df['author_clean']

y = train_df['general_category']
X = train_df.drop('general_category', axis=1, inplace=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=5)
allcats = sorted(list(set(train_df['general_category'])))

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

# below code adapted from:
# https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
# https://github.com/huggingface/transformers/blob/e6cff60b4cbc1158fbd6e4a1c3afda8dc224f566/examples/run_glue.py

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

MAX_LEN = 256
device = "cuda:0"
# In[6]:


def preprocessing_for_scibert(data):

    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(text=get_clean(sent),add_special_tokens=True,
                                             max_length=MAX_LEN, pad_to_max_length=True,return_attention_mask=True)

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





# In[9]:


train_inputs, train_masks = preprocessing_for_scibert(X_train['trainingText'])
val_inputs, val_masks = preprocessing_for_scibert(X_val['trainingText'])


# In[10]:


# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# num_training = y_train.shape[0]
# num_training
# allY = np.concatenate((y_train,y_val))

# label_encoder = LabelEncoder()
# int_encoded_Y = label_encoder.fit_transform(allY)

# onehot_encoder = OneHotEncoder(sparse=False)
# int_encoded_Y = int_encoded_Y.reshape(len(int_encoded_Y), 1)

# onehot_Y = onehot_encoder.fit_transform(int_encoded_Y)

# train_labels = torch.tensor(onehot_Y[:num_training])
# val_labels = torch.tensor(onehot_Y[num_training:])

# train_labels = torch.tensor(y_train_ind.values)
# val_labels = torch.tensor(y_val_ind.values)
# print ("train_labels", train_labels)
# print ("train_labels[0]", train_labels[0])
# print ("val_labels", val_labels)
# print ("val_labels[0]", val_labels[0])
# max_label = torch.cat((train_labels,val_labels)).max()
# max_label


# In[11]:


# print ("train_inputs", train_inputs[0])
# print ("train_masks", train_masks[0])
# print ("train_labels[0]", train_labels[0])


# In[12]:


batch_size = 8

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# In[13]:


# print ("max_label", max_label)
model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=labels_len,output_attentions=False,
                                                      output_hidden_states=False)

model.cuda()


# In[14]:


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

training_stats = []

total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()


        output = model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask,labels=b_labels)
        loss, logits = output[:2]

        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)

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
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,labels=b_labels.long())

        loss, logits = output[:2]
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

torch.save(model, 'model_arxiv_1nn.pt')
print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats.to_csv("training_stats_1nn.csv")
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

plt.savefig("scibert+1nn_epoch4_batch8.png")


# In[19]:



device = "cpu"
test_df = pd.read_csv("test.csv", index_col=0,nrows = 10)
test_df = test_df.replace(np.nan, '', regex=True)
test_df.shape
test_df.head()
# train_df = train_df.dropna(subset=['title', 'abstract', 'general_category'])
test_df["trainingText"] = test_df['title']+test_df['abstract']+test_df['author_clean']

y_test = test_df['general_category']
X_test = test_df.drop('general_category', axis=1, inplace=False)
y_test_ind = y_test.apply(lambda x: allcats.index(x))
test_labels = torch.tensor(y_test_ind.tolist())

test_inputs, test_masks = preprocessing_for_scibert(X_test['trainingText'])

batch_size = 8

prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
prediction_dataloader = DataLoader(prediction_data, sampler=SequentialSampler(prediction_data), batch_size=batch_size)

model = torch.load('model_arxiv_1nn.pt', map_location=torch.device('cpu'))
model.eval()
# Tracking variables
# Tracking variables
predictions , true_labels = [], []


for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask , b_labels = batch

    with torch.no_grad():
        logits = model(b_input_ids,attention_mask=b_input_mask)
    logits = logits[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

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
matthews_df.to_csv("matthews_1nn.csv")

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions,axis = 1).flatten()
flat_true_labels = np.concatenate(true_labels, axis=0)

dfpred = pd.DataFrame(flat_predictions)
dfpred.to_csv("predictions_1nn.csv")
cr = sklearn.metrics.classification_report(flat_true_labels, flat_predictions)
cm = np.array2string(sklearn.metrics.confusion_matrix(flat_true_labels, flat_predictions))
f = open('report_1nn.txt', 'w')
f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
f.close()

from sklearn.metrics import accuracy_score, hamming_loss
print('Hamming loss : {}'.format(hamming_loss(flat_true_labels, flat_predictions)))
print("Accuracy_score : {}".format(accuracy_score(flat_true_labels, flat_predictions)))


# # In[ ]:




