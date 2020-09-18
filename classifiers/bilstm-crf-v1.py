import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import convert_data_to_lstm as cdtl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

torch.manual_seed(1)


def argmax(vector):
    # return argmax of the vector
    _, idx = torch.max(vector, 1)
    return idx.item()


def log_sum_exp(vector):
    max_score = vector[0, argmax(vector)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vector.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vector - max_score_broadcast)))


class biLSTM_CRF(nn.Module):
    def __init__(self, class_to_ix, features_dim, hidden_dim):
        super(biLSTM_CRF, self).__init__()
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.class_to_ix = class_to_ix
        self.target_size = len(class_to_ix)
        # biLSTM init
        self.lstm = nn.LSTM(features_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=False, dropout=0.5)
        # maps the output of the lstm into the classes space
        self.tanh = nn.Tanh()
        self.hidden_to_class = nn.Linear(hidden_dim, self.target_size)
        # matrix of transition parameters - index i,j is the score of i->j
        self.transitions = nn.Parameter(
            torch.randn(self.target_size, self.target_size))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(1, 1, self.hidden_dim // 2),
                torch.randn(1, 1, self.hidden_dim // 2))

    def _forward_alg(self, features):
        init_alphas = torch.full(
            (1, self.target_size), 0, dtype=torch.long)

        # wrap variable for auto backprop
        forward_var = init_alphas

        for feature in features:
            alphas_t = []
            for next_class in range(self.target_size):
                emission_score = feature[next_class].view(
                    1, -1).expand(1, self.target_size)
                transition_score = self.transitions[next_class].view(1, -1)
                next_class_var = forward_var + transition_score + emission_score
                alphas_t.append(log_sum_exp(next_class_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        alpha = log_sum_exp(forward_var)
        return alpha

    def _get_lstm_features(self, sequence):
        self.hidden = self.init_hidden()
        lstm_out = lstm_out.view(len(sequence), self.hidden_dim)
        lstm_features = self.hidden_to_class(lstm_out)
        return lstm_features

    def _score_sequence(self, features, classes):
        # gives score of provided class sequence
        score = torch.zeros(1)
        for idx, feature in enumerate(features):
            if idx != len(features) - 1:
                score = score + self.transitions[classes[idx + 1],
                                                 classes[idx]] + feature[classes[idx + 1]]
        return score

    def _viterbi_decode(self, features):
        backpointers = []

        init_vitvars = torch.full(
            (1, self.target_size), 0, dtype=torch.long)  # change from -10000 to 0 and see what happens
        forward_var = init_vitvars
        for feature in features:
            bps_t = []  # backpointers to this step
            viterbivars_t = []  # viterbi variables for this step

            for next_class in range(self.target_size):
                next_class_var = forward_var + self.transitions[next_class]
                best_class_id = argmax(next_class_var)
                bps_t.append(best_class_id)
                viterbivars_t.append(
                    next_class_var[0][best_class_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feature).view(1, -1)
            backpointers.append(bps_t)
        terminal_var = forward_var
        best_class_id = argmax(terminal_var)
        path_score = terminal_var[0][best_class_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_class_id]
        for bptrs_t in reversed(backpointers):
            best_class_id = bptrs_t[best_class_id]
            best_path.append(best_class_id)
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sequence, classes):
        features = self._get_lstm_features(sequence)
        forward_score = self._forward_alg(features)
        gold_score = self._score_sequence(features, classes)
        return forward_score - gold_score

    def forward(self, sequence):
        # Get the emission scores from the BiLSTM
        lstm_features = self._get_lstm_features(sequence)
        # Find the best path, given the features.
        lstm_features = self.tanh(lstm_features)
        score, class_seq = self._viterbi_decode(lstm_features)
        return score, class_seq


def main():
    #####################################################################
    # data loader and manipulation:
    #####################################################################
    covid = cdtl.load_data()
    HIDDEN_DIM = 64
    FEATURES_DIM = len(covid.drop(
        [' Country', 'Date', 'target'], axis=1).columns)
    countries = cdtl.get_country_names(covid)
    training_data = cdtl.data_to_lstm_input(
        covid, countries, '2020-03-01', '2020-06-01')
    classes_to_ix = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    #####################################################################
    # initialize model and optimizer:
    #####################################################################
    model = biLSTM_CRF(classes_to_ix, FEATURES_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    print(
        f'Number of parameters: {sum(param.numel() for param in model.parameters())}')
    print(
        f'Num of trainable parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    #####################################################################
    # test model before training:
    #####################################################################
    with torch.no_grad():
        prechecks = [data[0] for data in training_data['Israel']]
        prechecks = torch.tensor(prechecks, dtype=torch.float)
        print(prechecks.shape)
        for idx, precheck in enumerate(prechecks):
            print('check:', model(precheck))
    #####################################################################
    # training:
    #####################################################################
    model.train()
    loss_values = []
    for epoch in range(
            150):
        count = 0
        total_epoch_loss = 0
        print('Number of epoch: {}'.format(epoch))
        for sequence, classes in training_data['Israel']:
            model.zero_grad()
            sequence = torch.tensor(sequence, dtype=torch.float)
            classes = torch.tensor(classes, dtype=torch.long)
            loss = model.neg_log_likelihood(sequence, classes)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            count += 1
        print(f'Average loss for epoch {epoch}: {total_epoch_loss/count}')
        loss_values.append(total_epoch_loss/count)
    plt.plot(loss_values)
    plt.show()
    #####################################################################
    # test training:
    #####################################################################
    with torch.no_grad():
        prechecks = [data[0] for data in training_data['Israel']]
        prechecks = torch.tensor(prechecks, dtype=torch.float)
        for idx, precheck in enumerate(prechecks):
            print('check:', model(precheck))


if __name__ == "__main__":
    main()
