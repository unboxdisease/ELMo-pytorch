import torch.nn.functional as F
import torch.nn as nn 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset import Dataset_seq,build_vocab

class Bi_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=11, num_layers=2):
        super(Bi_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers


        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)


    def forward(self, input):
        #Forward pass through initial hidden layer
        

        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, input_size ,hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (batch_size, num_layers, hidden_dim).
        lstm_out, self.hidden = self.lstm(input)
        
        q = torch.zeros([lstm_out.shape[0],1,100],device = 'cuda')
        n1= lstm_out[:,0:lstm_out.shape[1]-1,0:100] 
#         print(n1.shape,q.shape)
        n1 = torch.cat((q,n1),1)
        n2= lstm_out[:,1:lstm_out.shape[1],100:200]
        n2 = torch.cat((n2,q),1)
        
        new = torch.cat((n1,n2),2)
#         print(n1.shape,n2.shape,new.shape)

        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(new)
        return y_pred, lstm_out

class classifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim , 50)
        self.linear2 = nn.Linear(50, 5)
        


    def forward(self, input):
        #Forward pass through initial hidden layer
        lstm_out, (hidden,_) = self.lstm(input)
        
        y_p = self.linear(hidden)
        yp = F.relu(y_p)
        y_pred = self.linear2(yp)
        return y_pred