# contains model tobe used
import torch
import torch.nn as nn
from torch.nn.modules import dropout

class ourmodel(nn.Module):
    def __init__(self,vocab_len_of_robot_lang,embedding_dim,dropout_value,hidden_dim,device='cpu',n_layers=2):
        super().__init__()
        self.input_dim = vocab_len_of_robot_lang
        self.embedding_dim = embedding_dim
        self.dropout = dropout_value
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layers= n_layers
        self.embedding_layer = nn.Embedding(vocab_len_of_robot_lang,embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,hidden_dim,n_layers,dropout=dropout_value)
        self.dropout = nn.Dropout(dropout_value)


    def forward():
        pass
