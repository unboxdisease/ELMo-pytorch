#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset import Dataset_seq,build_vocab

import torch
import torch.nn as nn
from torch import optim
import time
from model import Bi_RNN

import wandb

model = Bi_RNN(50, 100, 100, 9773, 2)
model.load_state_dict(torch.load('./models/model_20.pt'))
model.to('cuda')
model.eval()


# In[2]:



train_path = "./Dataset/yelp-subset.train.csv"
test_path = "./Dataset/yelp-subset.test.csv"
word2id_ts,id2word_ts = build_vocab(train_path)
bs = 100
yelp_test = Dataset_seq(word2id_ts,id2word_ts, test_path)

test_dl = DataLoader(yelp_test, shuffle=True, batch_size=bs, num_workers=2)


# In[14]:


device = 'cuda'
criterion = torch.nn.CrossEntropyLoss()
val_loss = 0
val_accuracy = 0
val_samples = 0
for input_vector, label in test_dl:

        # add the input vector and label to the gpu
        input_vector = input_vector.float()
        input_vector = input_vector.to(device)
        label = label.to(device)

        # forward pass
        logits,_ = model(input_vector)
        logits = logits.view(logits.shape[1]* logits.shape[0], logits.shape[2])
        label = label.view(-1)
        

        
        # compute loss
        loss = criterion(logits, label)

        val_loss += loss.item()
        predictions = logits.argmax(dim=1)
        
        # update accuracy
        val_accuracy += (predictions == label).sum()

        val_samples += logits.shape[0]
        
        print("Progress : {0} ".format((val_samples/2000) * 100 / len(test_dl)),end='\r')
print((val_accuracy/val_samples).item()*100)


# In[ ]:




