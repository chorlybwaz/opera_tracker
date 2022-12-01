import torch
import torch.nn as nn


class LSTM_for_BCELoss_hidden(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, output_size, num_layers, dropout=0):
        super(LSTM_for_BCELoss_hidden, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout = self.dropout)

        # setup output layer
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden):
        lstm_out, hidden = self.lstm(input, hidden)
        out = self.linear(lstm_out.data)
        out = torch.sigmoid(out[:, 0])

        return out, hidden