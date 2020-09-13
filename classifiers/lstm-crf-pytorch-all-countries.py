#!/usr/bin/env python
# coding: utf-8

# In[73]:


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


# In[74]:


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


# In[75]:


def calc_accuracy(gold, predictions):
    count = 0
    for idx in range(len(gold)):
        if gold[idx] == predictions[0][idx]:
            count += 1
    return count


# In[76]:


def predict_sequences(training_data, test_data, model):
    with torch.no_grad():
        prechecks = [data[0] for data in training_data]
        labels = [data[1] for data in training_data]
        prechecks = torch.tensor(prechecks, dtype=torch.float)
        total_correct = 0
        total_predictions = len(prechecks)
        for idx, precheck in enumerate(prechecks):
            outputs, _ = model(precheck)
            #print('pred labels: ', outputs)
            #print('labels: ', labels[idx])
            total_correct += calc_accuracy(labels[idx], outputs)
        train_acc = total_correct/(total_predictions * 7)
        print(
            f'model accuracy after training: {train_acc}')
    with torch.no_grad():
        prechecks = [data[0] for data in test_data]
        labels = [data[1] for data in test_data]
        total_correct = 0
        total_predictions = len(prechecks)
        prechecks = torch.tensor(prechecks, dtype=torch.float)
        for idx, precheck in enumerate(prechecks):
            outputs, _ = model(precheck)
            #print('pred labels: ', outputs)
            #print('labels: ', labels[idx])
            total_correct += calc_accuracy(labels[idx], outputs)
        test_acc = total_correct/(total_predictions * 7)
        print(
            f'model accuracy after test: {test_acc}')
    return train_acc, test_acc


# In[77]:


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
        covid, countries, '2020-03-05', '2020-07-16')
    test_data = cdtl.data_to_lstm_input(
        covid, countries, '2020-07-16', '2020-08-13')
    # classes_to_ix = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    classes_to_ix = {0: 0, 1: 1, 2: 2, 3: 3}
    num_tags = 4
    #####################################################################
    # initialize model and optimizer:
    #####################################################################
    crf = CRF(num_tags)
    model = biLSTM(classes_to_ix, FEATURES_DIM, HIDDEN_DIM, crf)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    accumulated_grad_steps = 1
    #####################################################################
    # training:
    #####################################################################
    model.train()
    total_train_acc_values = []
    total_test_acc_values = []
    epoch_loss = []
    best_test_acc = 0.62
    for epoch in range(
            500):
        count = 0
        total_epoch_loss = 0
        i = 0
        loss_values = []
        train_acc_values = []
        test_acc_values = []
        print('Number of epoch: {}'.format(epoch))
        for country in countries:
            print(f'Training on {country}')
            for sequence, classes in training_data[country]:
                i += 1
                #model.zero_grad()
                sequence = torch.tensor(sequence, dtype=torch.float)
                classes = torch.tensor(classes, dtype=torch.long)
                _, lstm_features = model(sequence)
                loss = criterion(lstm_features, classes)
                loss = loss / accumulated_grad_steps
                loss.backward()

                if i % accumulated_grad_steps == 0:
                    optimizer.step()
                    model.zero_grad()

                total_epoch_loss += loss.item()
                count += 1
            print(f'Average loss for epoch {epoch}: {total_epoch_loss/count}')
            loss_values.append(total_epoch_loss/count)

            train_acc, test_acc = predict_sequences(training_data[country], test_data[country], model)
            train_acc_values.append(train_acc)
            test_acc_values.append(test_acc) 
        total_loss = sum(loss_values) / len(countries)
        total_train_acc = sum(train_acc_values) / len(countries)
        total_test_acc = sum(test_acc_values) / len(countries)
        epoch_loss.append(total_loss)
        total_train_acc_values.append(total_train_acc)
        total_test_acc_values.append(total_test_acc)
        if total_test_acc > best_test_acc:
            best_test_acc = total_test_acc
            torch.save(model.state_dict(), f'pickle_one_month_world_model_{total_train_acc}_train_acc_{best_test_acc}_test_acc_500_epoch.pkl')
            torch.save(model.state_dict(), f'one_month_world_model_{total_train_acc}_train_acc_{best_test_acc}_test_acc_500_epoch.pth')
            torch.save(model, f'entire_model_one_month_world_{total_train_acc}_train_acc_{best_test_acc}_test_acc_500_epoch.pth')
    plt.plot(total_train_acc_values)
    plt.plot(total_test_acc_values)
    plt.show()
    plt.plot(epoch_loss)
    plt.show()


# In[78]:


if __name__ == "__main__":
    main()

