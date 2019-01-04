import matplotlib

matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Net2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, 120), nn.Dropout(0.1))
        self.dense1_bn = nn.BatchNorm1d(120)
        self.fc1 = nn.Linear(120, 84)
        self.dense2_bn = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.dense1_bn(self.fc(out[:, -1, :]))

        x = F.relu(self.dense2_bn(self.fc1(out)))

        x = (self.fc2(x))
        return x
