from sklearn.datasets import fetch_20newsgroups
from torch import nn
import torch
import numpy as np
import torch.functional as F

from torch import nn
import torch
import numpy as np
import torch.functional as F
from torch.nn.utils import rnn as rnn_utils


class JobtypeClassifier_FeedForward(nn.Module):
    def __init__(self, num_features, label_nums=2):
        super(JobtypeClassifier_FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=label_nums)
    def forward(self, x_in, apply_sigmoid=False):
        x_in = x_in.view(x_in.shape[0], -1)
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out



import torch.functional as F

class JobtypeClassifier_Conv1d(nn.Module):
    def __init__(self, embedding_size, num_channels, hidden_dim, label_nums, dropout_p):

        super(JobtypeClassifier_Conv1d, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
            out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2, padding=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2,  padding=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU()
            )



        if num_channels == 128:
            max_pool_size = 31 
        elif num_channels == 10:
            max_pool_size = 2
        else:
            max_pool_size = None

        self.pool = nn.MaxPool1d(max_pool_size)
        self.rlue = nn.ReLU()
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, label_nums)
        self.drop1 = nn.Dropout(dropout_p)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x_in, apply_softmax=False):

        if len(x_in.shape) < 3:
            features = x_in
        else:
            x_embedded = x_in.permute(0, 2, 1)
            features = self.convnet(x_embedded)
            features = self.pool(features).squeeze(dim=2)

        
        features = self.drop1(features)
        features = self.fc1(features)
        features = self.drop2(features)
        features = self.rlue(features)
        prediction_vector = self.fc2(features).squeeze(dim=1)
        if apply_softmax:
             prediction_vector = F.softmax(prediction_vector, dim=1)
             
        return prediction_vector





class RnnModel1(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(RnnModel1, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None
        # Simple RNN
        self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first=True)
        # Classifier to produce as many logits as outputs
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)
    def forward(self, X):
        # X is batch first (N, L, F)
        # output is (N, L, H)
        # final hidden state is (1, N, H)
        batch_first_output, self.hidden = self.basic_rnn(X)
        # only last item in sequence (N, 1, H)
        last_output = batch_first_output[:, -1]
        # classifier will output (N, 1, n_outputs)
        out = self.classifier(last_output)
        # final output is (N, n_outputs)
        return out.view(-1, self.n_outputs)



class ElmanRNN(nn.Module):
    """ an Elman RNN built using the RNNCell """
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(ElmanRNN, self).__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.batch_first = batch_first
        self.hidden_size = hidden_size
    def _initial_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, x_in, initial_hidden=None):

        if self.batch_first:
            batch_size, seq_size, feat_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feat_size = x_in.size()
            
        hiddens = []
        if initial_hidden is None:
            initial_hidden = self._initial_hidden(batch_size)
            initial_hidden = initial_hidden.to(x_in.device)

        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hiddens.append(hidden_t)

        hiddens = torch.stack(hiddens)
        if self.batch_first:
            hiddens = hiddens.permute(1, 0, 2)
        return hiddens




class RnnModel2(nn.Module):
    """ A Classifier with an RNN to extract features and an MLP to classify """
    def __init__(self, embedding_size, num_embeddings, num_classes, rnn_hidden_size, batch_first=True, padding_idx=0):
        super(RnnModel2, self).__init__()

        self.rnn = ElmanRNN(input_size=embedding_size, hidden_size=rnn_hidden_size, batch_first=batch_first)
        self.fc1 = nn.Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x_in, x_lengths=None, apply_softmax=False):

        y_out = self.rnn(x_in)
        y_out = y_out[:, -1, :]

        y_out = self.drop1(y_out)
        y_out = self.fc1(y_out)
        y_out = self.relu(y_out)

        y_out = self.drop2(y_out)
        y_out = self.fc2(y_out)
        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
        return y_out





class LstmModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(LstmModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None
        self.cell = None
        # Simple LSTM
        self.basic_rnn = nn.LSTM(self.n_features, self.hidden_dim, bidirectional=True)
        # Classifier to produce as many logits as outputs
        self.classifier = nn.Linear(2 * self.hidden_dim, self.n_outputs)

    def forward(self, X):
        lengths = [X.shape[1]] * X.shape[0]
        X_pack = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        rnn_out, (self.hidden, self.cell) = self.basic_rnn(X_pack)
        # unpack the output (N, L, 2*H)
        batch_first_output, seq_sizes = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
        # only last item in sequence (N, 1, 2*H)
        seq_idx = torch.arange(seq_sizes.size(0))
        last_output = batch_first_output[seq_idx, seq_sizes-1]
        # classifier will output (N, 1, n_outputs)
        out = self.classifier(last_output)
        # final output is (N, n_outputs)
        return out.view(-1, self.n_outputs)