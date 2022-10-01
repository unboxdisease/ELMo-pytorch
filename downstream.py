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
from model import Bi_RNN,classifier

train_path = './Dataset/yelp-subset.train.csv'
word2id_ts,id2word_ts = build_vocab(train_path)
model = Bi_RNN(50, 100, 100, 9773, 2)
model.load_state_dict(torch.load('./models/model_20.pt'))
model.to('cuda')
model.eval()


# In[2]:


import pandas as pd
import gensim
from dataset import Dataset_seq,Dataset_class
yelp_train = Dataset_seq(word2id_ts,id2word_ts, train_path)
word2rep = yelp_train.word2representation
# def csv_to_dataframe(path):
df = pd.read_csv(train_path, sep=',', header=None)
df.columns = ['label', 'text']
df = df[1:]
X = []
Y = []
for idx, row in enumerate(df.text):
    sent = gensim.utils.simple_preprocess(df.iloc[idx].text,min_len=1)
    sent = [word if word in yelp_train.word2id else '<unk>' for word in sent]
    X.append([word2rep[w] for w in sent])
    Y.append(df.iloc[idx].label)


# In[5]:


X1 = []
for i in range(1,len(X)) : 
    b = torch.FloatTensor(np.array([X[i]])).to('cuda')
    out,lstm = model(b)
    lstm = lstm.view(lstm.shape[0]*lstm.shape[1],lstm.shape[2])
    X1.append(lstm)


# In[ ]:



wandb.init(project="ELMo", entity="kushaljain")
lr = 0.0003
epochs = 50
criterion = torch.nn.CrossEntropyLoss()
model = model.float()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
error = {"train": [], "val": []}
accuracy = {"train" : [], "val" : []}
perplexity = {"train": [], "val": []}
times = {"train": [], "val": []}
wandb.config = {
  "learning_rate": 0.003,
  "epochs": 100,
  "batch_size": 100
}
print(device)

import sys

for epoch in range(epochs):
    t1 = time.time()
    print("starting Epoch : " + str(epoch + 1))

    train_samples = 0
    val_samples = 0
    train_loss = 0
    val_loss = 0
    val_accuracy = 0
    train_accuracy = 0
    t1 = time.time()
    model.train()
    for input_vector, label in train_dl:
        

        # add the input vector and label to the gpu
        input_vector = input_vector.float()
        input_vector = input_vector.to(device)
        label = label.to(device)

        # forward pass
        logits,_ = model(input_vector)
#         print("shape of the logits : {0}",format(logits.shape))
#         print("shape of the label : {0}",format(label.shape))       
        logits = logits.view(logits.shape[1]* logits.shape[0], logits.shape[2])
        label = label.view(-1)

        # compute loss
        loss = criterion(logits, label)

        # set the gradients to zero
        optimizer.zero_grad()

        # back.prop
        loss.backward()

        # update parameters
        optimizer.step()

        train_loss += loss.item()
        predictions = logits.argmax(dim=1)

        # update accuracy
        train_accuracy += (predictions == label).sum()  
        train_samples += logits.shape[0]
#         print(train_accuracy)
#         print(train_samples)
        print("Progress : {0} ".format((train_samples/2000) * 100 / len(train_dl)),end='\r')
        
    t2 = time.time()
    print("Time taken to run training for epoch {0} : {1} ".format(
        epoch, t2 - t1))
    t3 = time.time()

    model.eval()
    for input_vector, label in val_dl:

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
        
        print("Progress : {0} ".format((val_samples/2000) * 100 / len(val_dl)),end='\r')
        

    t4 = time.time()
    train_perplexity = np.exp(train_loss/len(train_dl))
    val_perplexity = np.exp(val_loss/len(val_dl))
    train_accuracy = train_accuracy / train_samples
    val_accuracy = val_accuracy / val_samples
    print("Time taken to run training for epoch {0} : {1} ".format(
        epoch, t4 - t3))
    print("Total Time for epoch {0} is {1}".format(epoch+1, t4 - t1))
    print("Training Loss for epoch {0} : {1} ".format(epoch + 1, train_loss))
    print("Training Accuracy for epoch {0} : {1}".format(
        epoch + 1, train_accuracy))

    print("Val Loss for epoch {0} : {1} ".format(epoch + 1, val_loss))
    print("Val Accuracy for epoch {0} : {1}".format(epoch + 1, val_accuracy))
    print("Train perplexity for epoch {0} : {1}".format(
        epoch + 1, train_perplexity))
    print("Val perplexity for epoch {0} : {1}".format(
        epoch + 1, val_perplexity))
    error['train'].append(train_loss)
    error['val'].append(val_loss)
    accuracy['train'].append(train_accuracy)
    accuracy['val'].append(val_accuracy)
    times['train'].append(t2 - t1)
    times['val'].append(t4-t3)
    perplexity['train'].append(train_perplexity)
    perplexity['val'].append(val_perplexity)

    
    wandb.log({
        "Training loss": train_loss,
        "Validation loss": val_loss,
        "Train accuracy" : train_accuracy,
        "Validation accuracy" : val_accuracy
              })
    
    # save the model
    if (epoch + 1) % 5 == 0:
        print("Saving the model ")
        torch.save(model.state_dict(),'./models/model_{0}.pt'.format(epoch + 1))
wandb.finish()

