import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import convert_data_to_lstm as cdtl
import pandas as pd
import numpy as np

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
                            num_layers=1, bidirectional=True, dropout=0.5)
        # maps the output of the lstm into the classes space
        self.hidden_to_class = nn.Linear(hidden_dim, self.target_size)
        # matrix of transition parameters - index i,j is the score of i->j
        self.transitions = nn.Parameter(
            torch.rand(self.target_size, self.target_size))  # check with rand instead randn
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, features):
        init_alphas = torch.full(
            (1, self.target_size), 0, dtype=torch.long)  # change from -10000 to 0 and see what happens

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
        # sequence = torch.FloatTensor(sequence).view(len(sequence), 1, -1)  # change from FloatTensor to LongTensor
        sequence = sequence.view(len(sequence), 1, -1)
        lstm_out, self.hidden = self.lstm(sequence, self.hidden)
        lstm_out = lstm_out.view(len(sequence), self.hidden_dim)
        lstm_features = self.hidden_to_class(lstm_out)
        return lstm_features

    def _score_sequence(self, features, classes):
        # gives score of provided class sequence
        score = torch.zeros(1)
        for idx, feature in enumerate(features):
            if idx != len(features) - 1:
                score = score + self.transitions[classes[idx + 1],
                                                 classes[idx]] + feature[classes[idx + 1]]  # change from
        return score

    def _viterbi_decode(self, features):  # problem in viterbi?
        backpointers = []

        init_vitvars = torch.full(
            (1, self.target_size), 0, dtype=torch.long)  # change from -10000 to 0 and see what happens
        # init_vitvars[0][self.class_to_ix[0]] = 0 not sure what this stage does
        forward_var = init_vitvars
        # print('forward: ', forward_var)
        # print('transition: ', self.transitions)
        for feature in features:
            #print('feature=', feature)
            bps_t = []  # backpointers to this step
            viterbivars_t = []  # viterbi variables for this step

            for next_class in range(self.target_size):
                #print('next class: ', next_class)
                next_class_var = forward_var + self.transitions[next_class]
                #print('self transition: ', self.transitions)
                #print('transition: ', self.transitions[next_class])
                #print('var: ', next_class_var)
                # print('next class var: ', next_class_var)
                best_class_id = argmax(next_class_var)
                bps_t.append(best_class_id)
                viterbivars_t.append(
                    next_class_var[0][best_class_id].view(1))  # check with abs
            # print('viterbis: ', viterbivars_t)
            forward_var = (torch.cat(viterbivars_t) + feature).view(1, -1)
            backpointers.append(bps_t)
        print('backpointers: ', backpointers)
        terminal_var = forward_var
        #print('terminal var: ', terminal_var)
        best_class_id = argmax(terminal_var)
        # print(terminal_var)
        path_score = terminal_var[0][best_class_id]
        #print('path score: ', path_score)

        # Follow the back pointers to decode the best path.
        best_path = [best_class_id]
        #best_path = []
        for bptrs_t in reversed(backpointers):
            best_class_id = bptrs_t[best_class_id]
            #print('best tag id: ', best_class_id)
            best_path.append(best_class_id)
            #print('best path', best_path)
        best_path.reverse()
        #print('best path: ', best_path)
        return path_score, best_path

    def neg_log_likelihood(self, sequence, classes):  # seems loss is quite okay
        features = self._get_lstm_features(sequence)
        # print('features: ', features)
        # print('##########################################')
        forward_score = self._forward_alg(features)
        gold_score = self._score_sequence(features, classes)
        #print('forward, gold: ', forward_score, gold_score)
        #print('forward - gold : ', forward_score - gold_score)
        return forward_score - gold_score

    def forward(self, sequence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_features = self._get_lstm_features(sequence)
        # Find the best path, given the features.
        score, class_seq = self._viterbi_decode(lstm_features)
        return score, class_seq


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
    classes_to_ix = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    #####################################################################
    # initialize model and optimizer:
    #####################################################################
    model = biLSTM_CRF(classes_to_ix, FEATURES_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    #####################################################################
    # test model before training:
    #####################################################################
    with torch.no_grad():
        prechecks = [data[0] for data in training_data['Israel']]
        prechecks = torch.tensor(prechecks, dtype=torch.float)
        for idx, precheck in enumerate(prechecks):
            print('check:', model(precheck))
    #####################################################################
    # training:
    #####################################################################
    for epoch in range(
            5):
        print('Number of epoch: {}'.format(epoch))
        for sequence, classes in training_data['Israel']:
            model.zero_grad()
            sequence = torch.tensor(sequence, dtype=torch.float)
            classes = torch.tensor(classes, dtype=torch.long)
            loss = model.neg_log_likelihood(sequence, classes)
            #print('loss={} in epoch={}'.format(loss, epoch))
            loss.backward()
            optimizer.step()
        # for name, p in model.named_parameters():
        #    if name == 'transitions':
        #        print(name, p.data)
    #####################################################################
    # test training:
    #####################################################################
    with torch.no_grad():
        prechecks = [data[0] for data in training_data['Israel']]
        prechecks = torch.tensor(prechecks, dtype=torch.float)
        for idx, precheck in enumerate(prechecks):
            print('check:', model(precheck))
            # print(training_data['Israel'][idx][1])"""
    # with torch.no_grad():
    #    prechecks = [training_data['Israel'][0]
    #                 [0], training_data['Israel'][1][0]]
    #    prechecks = torch.tensor(prechecks, dtype=torch.float)
    #    for precheck in prechecks:
    #        print('check:', model(precheck))


if __name__ == "__main__":
    main()
