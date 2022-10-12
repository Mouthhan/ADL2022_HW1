from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from torchinfo import summary

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, class_num):
        super(LSTM, self).__init__()
        # pass argument
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.class_num = class_num
        # initial LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            layer_num, dropout=0.2, bidirectional=True,batch_first=True)
        self.gru = nn.GRU(input_dim, hidden_dim,
                            layer_num, dropout=0.2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.gelu = nn.GELU()
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, class_num)
        )
        # self.linear = nn.Linear(hidden_dim, class_num)

    def forward(self, x):
        x = self.ln(x)
        output, hn = self.gru(x)
        # output, (hn, cn) = self.lstm(x, None)
        feature = self.bn(hn[-1])
        feature = self.gelu(feature)
        feature = self.dropout(feature)
        result = self.linear(feature)
        return result

class LSTM_Tagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num, class_num,max_len):
        super(LSTM_Tagger, self).__init__()
        # pass argument
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.class_num = class_num
        self.max_len = max_len
        # initial LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            layer_num, dropout=0.2, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(input_dim, hidden_dim,
                            layer_num, dropout=0.2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(max_len)
        self.ln = nn.LayerNorm(input_dim)
        # self.linear = nn.Linear(2 * hidden_dim, class_num)
        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_dim,2 * hidden_dim),
            nn.BatchNorm1d(max_len),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim,2 * hidden_dim),
            nn.BatchNorm1d(max_len),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, class_num)
        )

    def forward(self, x):
        x = self.ln(x)
        # output, hn = self.gru(x)
        output, (hn, cn) = self.lstm(x)
        output = self.bn(output)
        output = self.gelu(output)
        output = self.dropout(output)
        # tag_seq = self.linear(output)
        # print(output.shape)
        tag_seq = self.linear(output.view(len(x), self.max_len, -1))
        return tag_seq

# model = LSTM(300, 1024, 2, 9).cuda()
# model = LSTM_Tagger(300, 1024, 2, 9, 32).cuda()
# summary(model, input_size=(16, 32, 300))
# class SeqClassifier(torch.nn.Module):
#     def __init__(
#         self,
#         embeddings: torch.tensor,
#         hidden_size: int,
#         num_layers: int,
#         dropout: float,
#         bidirectional: bool,
#         num_class: int,
#     ) -> None:
#         super(SeqClassifier, self).__init__()
#         self.embed = Embedding.from_pretrained(embeddings, freeze=False)
#         # TODO: model architecture

#     @property
#     def encoder_output_size(self) -> int:
#         # TODO: calculate the output dimension of rnn
#         raise NotImplementedError

#     def forward(self, batch) -> Dict[str, torch.Tensor]:
#         # TODO: implement model forward
#         raise NotImplementedError
