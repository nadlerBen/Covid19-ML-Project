from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import datetime
import warnings
warnings.filterwarnings(action='ignore')


def load_and_split(data):
    covid = pd.read_csv('project_data_preprocessed.csv')
    covid = covid.drop(['Unnamed: 0', 'Date_reported', ' Country'], axis=1)
    X = covid[covid.columns[:-1]].to_numpy()
    y = covid[covid.columns[-1]].to_numpy()
    return X, y
