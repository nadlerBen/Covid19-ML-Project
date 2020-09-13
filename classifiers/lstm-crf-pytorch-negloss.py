import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchcrf import CRF
import convert_data_to_lstm as cdtl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

torch.manual_seed(1)


def argmax(vector):
    # return argmax of the vector
    _, idx = torch.max(vector, 1)
    return idx


class biLSTM(nn.Module):
    def __init__(self, class_to_ix, features_dim, hidden_dim, crf):
        super(biLSTM, self).__init__()
        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.class_to_ix = class_to_ix
        self.target_size = len(class_to_ix)
        self.crf = crf
        # biLSTM init
        self.lstm = nn.LSTM(features_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        # maps the output of the lstm into the classes space
        self.hidden_to_class = nn.Linear(hidden_dim, self.target_size)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sequence):
        self.hidden = self.init_hidden()
        sequence = sequence.view(len(sequence), 1, -1)
        lstm_out, self.hidden = self.lstm(sequence, self.hidden)
        lstm_out = lstm_out.view(len(sequence), self.hidden_dim)
        lstm_features = self.hidden_to_class(lstm_out)
        # lstm_features = F.log_softmax(lstm_features, dim=1)
        return lstm_features

    def neg_log_likelihood(self, sequence, classes):
        features = self._get_lstm_features(sequence).view(7, 1, 4)
        loss = self.crf.forward(features, classes)
        return -1 * loss

    def forward(self, sequence):
        # Get the emission scores from the BiLSTM
        lstm_features = self._get_lstm_features(sequence)
        # Find the best path, given the features.
        lstm_features = lstm_features.view(7, 1, 4)
        class_seq = self.crf.decode(lstm_features)
        lstm_features = F.log_softmax(lstm_features.view(7, 4), dim=1)
        return class_seq, lstm_features


def calc_accuracy(gold, predictions):
    count = 0
    for idx in range(len(gold)):
        if gold[idx] == predictions[0][idx]:
            count += 1
    return count


def main():
    #####################################################################
    # data loader and manipulation:
    #####################################################################
    covid = cdtl.load_data()
    HIDDEN_DIM = 10
    FEATURES_DIM = len(covid.drop(
        [' Country', 'Date', 'target'], axis=1).columns)
    countries = cdtl.get_country_names(covid)
    training_data = cdtl.data_to_lstm_input(
        covid, countries, '2020-03-05', '2020-08-13')
    # classes_to_ix = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    classes_to_ix = {0: 0, 1: 1, 2: 2, 3: 3}
    num_tags = 4
    #####################################################################
    # initialize model and optimizer:
    #####################################################################
    crf = CRF(num_tags)
    model = biLSTM(classes_to_ix, FEATURES_DIM, HIDDEN_DIM, crf)
    criterion = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0004571863263754201,
                           weight_decay=4.53428947377099e-06, betas=(0.9756066728237405, 0.8520427545571541))
    #####################################################################
    # training:
    #####################################################################
    model.train()
    loss_values = []
    for epoch in range(
            250):
        count = 0
        total_epoch_loss = 0
        print('Number of epoch: {}'.format(epoch))
        for sequence, classes in training_data['Israel']:
            model.zero_grad()
            sequence = torch.tensor(sequence, dtype=torch.float)
            classes = torch.tensor(classes, dtype=torch.long).view(7, 1)
            loss = model.neg_log_likelihood(sequence, classes)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            count += 1
        print(f'Average loss for epoch {epoch}: {total_epoch_loss/count}')
        loss_values.append(total_epoch_loss/count)
        # for name, p in model.named_parameters():
        #    if name == 'transitions':
        #        print(name, p.data)
    plt.plot(loss_values)
    plt.show()
    #####################################################################
    # test training:
    #####################################################################
    with torch.no_grad():
        prechecks = [data[0] for data in training_data['Israel']]
        labels = [data[1] for data in training_data['Israel']]
        prechecks = torch.tensor(prechecks, dtype=torch.float)
        total_correct = 0
        total_predictions = len(prechecks)
        for idx, precheck in enumerate(prechecks):
            outputs, _ = model(precheck)
            # print('check:', outputs)
            print('pred labels: ', outputs)
            print('labels: ', labels[idx])
            total_correct += calc_accuracy(labels[idx], outputs)
        print(
            f'model accuracy after training: {total_correct/(total_predictions * 7)}')

    test_data = cdtl.data_to_lstm_input(
        covid, countries, '2020-03-05', '2020-08-13')
    with torch.no_grad():
        prechecks = [data[0] for data in test_data['Mexico']]
        labels = [data[1] for data in test_data['Mexico']]
        total_correct = 0
        total_predictions = len(prechecks)
        prechecks = torch.tensor(prechecks, dtype=torch.float)
        for idx, precheck in enumerate(prechecks):
            outputs, _ = model(precheck)
            print('pred labels: ', outputs)
            print('labels: ', labels[idx])
            total_correct += calc_accuracy(labels[idx], outputs)
        print(
            f'model accuracy after test: {total_correct/(total_predictions * 7)}')


if __name__ == "__main__":
    main()
