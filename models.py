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
            nn.Conv1d(self.input_channels, self.input_channels * 2, self.kernel_size, self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(self.input_channels * 2, self.input_channels * 4, self.kernel_size, self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(self.input_channels * 8, self.input_channels * 16, self.kernel_size, self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(self.input_channels * 16, self.input_channels * 32, self.kernel_size, self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(self.input_channels * 32, self.input_channels * 64, kernel_size = 1, stride = self.stride, padding_mode = "reflect"),
            nn.LeakyReLU(inplace = True),
            nn.Linear(self.input_channels * 64, self.output_channels * 64),
            nn.LeakyReLU(inplace = True),
            nn.Linear(self.output_channels * 64, self.output_channels * 8),
            nn.ReLU(inplace = True),
            nn.Linear(self.output_channels * 8, self.output_channels),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        
        out = self.main(input)
        return out
    

class lstm_generator(nn.Module):

    def __init__(self, input_size, batch_size, hidden_size, output_size):
        super(lstm_generator, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.main = nn.Sequential(
            nn.LSTM(self.input_size, self.hidden_size, 2, dropout = .2, bidirectional = True),
            nn.Tanh(),
            nn.LSTM(self.hidden_size * 2, self.hidden_size / 2, num_layers = 2, dropout = .2, bidirectional = True),
            nn.Tanh(),
            nn.LSTM(self.hidden_size, self.hidden_size / 4, 2, dropout = .2, bidirectional = True),
            nn.Tanh(),
            nn.Linear(self.hidden_size / 2, self.output_size * 4),
            nn.Linear(self.output_size * 4, self.output_size * 2),
            nn.Linear(self.hidden_size * 2, self.output_size)
        )

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
    

    def forward(self, input):

        out = self.main(input)
        return out