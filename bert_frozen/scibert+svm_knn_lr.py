import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import re
import random
import sklearn

# !pip install transformers
# !pip install allennlp
from transformers import *
from sklearn.model_selection import train_test_split


def get_clean(x):
    # Remove all characters not in the English alphabet
    x = re.sub(r"[^a-zA-Z0-9]+", ' ', x)
    x = str(x).lower()
    x = re.sub(r'\s+', ' ',x).strip()
    return x

train_df = pd.read_csv("train.csv", index_col=0, nrows=100000)
train_df = train_df.replace(np.nan, '', regex=True)
train_df.shape
train_df.head()
# train_df = train_df.dropna(subset=['title', 'abstract', 'general_category'])
train_df["trainingText"] = train_df['title']+train_df['abstract']+train_df['author_clean']
train_df["trainingText"] = train_df["trainingText"].apply(lambda x: get_clean(x))

y = train_df['general_category']
X_train, X_val, y_train, y_val = train_test_split(train_df["trainingText"], y, test_size=0.25, random_state=5)
# allcats = sorted(list(set(train_df['general_category'])))

y_train = y_train.values
y_val= y_val.values
print(type(y_val))

train_set = X_train.to_list()
val_set = X_val.to_list()

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', padding = True, do_lower_case = True,
                                          pad_token = "<pad>")
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)

train_out = []
for i in range(len(train_set)):
    ids = tokenizer.encode(train_set[i], add_special_tokens=True, truncation = True,max_length = 256 )

  # Convert the list of IDs to a tensor of IDs
    ids = torch.LongTensor(ids)

  # unsqueeze IDs to get batch size of 1 as added dimension
    ids = ids.unsqueeze(0)

    with torch.no_grad():
        out = model(input_ids=ids)

  # the output is a tuple
  # the tuple contains three elements as explained above)
  # we only want the hidden_states
    hidden_states = out[2]

    sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()

  # convert to numpy
    np_tensor_temp = sentence_embedding.numpy()
    train_out.append(np_tensor_temp)


val_out = []
for i in range(len(val_set)):
    ids = tokenizer.encode(val_set[i],add_special_tokens=True, truncation = True,max_length = 256)
    ids = torch.LongTensor(ids)
    ids = ids.unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=ids)
    hidden_states = out[2]
    sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
    np_tensor_temp = sentence_embedding.numpy()
    val_out.append(np_tensor_temp)



from numpy import savetxt

savetxt('x_train_from_bert_svm.csv', train_out, delimiter=',')
savetxt('x_val_from_bert_svm.csv', val_out, delimiter=',')


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score,hamming_loss
metrics = []

from sklearn.svm import SVC
print('SVC model fitting......')
clf_SVC = SVC().fit(train_out, y_train)
pred_svc = clf_SVC.predict(val_out)
acc_svc = accuracy_score(y_val, pred_svc)
hemm_svc = hamming_loss(y_val, pred_svc)

metrics.append(
    {
        "accuracy":acc_svc,
        "hemming":hemm_svc

    }
)
print("SVC accuracy: " + str(acc_svc))
print("SVC hemming: " + str(hemm_svc))

crsvc = sklearn.metrics.classification_report(y_val, pred_svc)
cmsvc = np.array2string(sklearn.metrics.confusion_matrix(y_val, pred_svc))
f = open('report_bert_svc.txt', 'w')
f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(crsvc, cmsvc))
f.close()


from sklearn.neighbors import KNeighborsClassifier
print('KNN model fitting......')
clf_knn = KNeighborsClassifier(n_neighbors=5).fit(train_out, y_train)
pred_knn = clf_knn.predict(val_out)
acc_knn = accuracy_score(y_val, pred_knn)
hemm_knn = hamming_loss(y_val, pred_knn)


metrics.append(
    {
        "accuracy":acc_knn,
        "hemming":hemm_knn

    }
)
print("SVC accuracy: " + str(acc_knn))
print("SVC hemming: " + str(hemm_knn))

crknn = sklearn.metrics.classification_report(y_val, pred_knn)
cmknn = np.array2string(sklearn.metrics.confusion_matrix(y_val, pred_knn))
f = open('report_bert_knn.txt', 'w')
f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(crknn, cmknn))
f.close()


from sklearn.linear_model import LogisticRegression
print('log reg model fitting......')
clf_LR = LogisticRegression(max_iter=5000).fit(train_out, y_train)
pred_LR = clf_LR.predict(val_out)
acc_LR = accuracy_score(y_val, pred_LR)
hemm_LR = hamming_loss(y_val, pred_LR)


metrics.append(
    {
        "accuracy":acc_LR,
        "hemming":hemm_LR

    }
)
print("SVC accuracy: " + str(acc_LR))
print("SVC hemming: " + str(hemm_LR))

crLR = sklearn.metrics.classification_report(y_val, pred_LR)
cmLR = np.array2string(sklearn.metrics.confusion_matrix(y_val, pred_LR))
f = open('report_bert_LR.txt', 'w')
f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(crLR, cmLR))
f.close()


df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv("metrics_svc_knn.csv")
