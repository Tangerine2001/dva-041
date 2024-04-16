import torch
from torch import nn


class cnn_discriminator(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super(cnn_discriminator, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.main = nn.Sequential(
            nn.Conv1d(self.input_channels, 15, self.kernel_size, self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(15, 30, self.kernel_size, self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(30, 60, self.kernel_size, self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(60, 120, self.kernel_size, self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(120, 1, kernel_size = 1, stride = self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Linear(11, self.output_channels * 220, bias = False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(self.output_channels * 220, self.output_channels * 220, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(self.output_channels * 220, self.output_channels * 1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        out = self.main(input.float())
        return out
    

class lstm_generator(nn.Module):

    def __init__(self, input_size, batch_size, hidden_size, output_size):
        super(lstm_generator, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size, 2, dropout = .2, bidirectional = False)
        self.tanh1 = nn.Tanh()
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, num_layers = 2, dropout = .2, bidirectional = False)
        self.tanh2 = nn.Tanh()
        self.lstm3 = nn.LSTM(self.hidden_size, self.hidden_size, 2, dropout = .2, bidirectional = False)
        self.tanh3 = nn.Tanh()
        self.linears = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.Linear(32, 16),
            nn.Linear(16, self.output_size)
        )
    

    def forward(self, input):

        lout1, (h1, c1) = self.lstm1(input.float())
        tout1 = self.tanh1(lout1)
        lout2, (h2, c2) = self.lstm2(tout1, (h1, c1))
        tout2 = self.tanh2(lout2)
        lout3, (h3, c3) = self.lstm3(tout2, (h2, c2))
        tout3 = self.tanh3(lout3)
        out = self.linears(tout3)

        return out