import torch.nn.functional as F
import torch.nn as nn 
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

        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out)
        return F.log_softmax(y_pred, dim=1), self.hidden