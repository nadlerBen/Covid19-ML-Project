import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler


def load_data():
    # changed from newer
    covid = pd.read_csv('newer_project_data_preprocessed.csv')
    covid['Date'] = pd.to_datetime(covid['Date_reported'])
    covid = covid.set_index(covid['Date'])
    covid = covid.drop(['Unnamed: 0', 'Date_reported'], axis=1)
    return covid


def get_week_range(start_date: str) -> str:
    end_date = pd.to_datetime(start_date) + pd.DateOffset(days=6)
    end_date = str(end_date).split()[0]
    return start_date, end_date


def get_week_vector(data, start_date: str, end_date: str):
    scaler = StandardScaler()
    week_data = data.loc[start_date:end_date]
    week_data = week_data.drop([' Country', 'Date'], axis=1)
    week_data[week_data.columns[:-1]
              ] = scaler.fit_transform(week_data[week_data.columns[:-1]])
    features = week_data[week_data.columns[:-1]].to_numpy().tolist()
    classes = week_data[week_data.columns[-1]].to_numpy().tolist()
    return (features, classes)

# wanted input format = [(7 day features, 7 day classes)]


def get_country_names(covid):
    return set(covid[' Country'])


def data_to_lstm_input(data, countries, start_date: str, end_date: str):
    input = {}
    date_format = '%Y-%m-%d'
    for country in countries:
        st = start_date
        en = start_date
        print('Working on {}'.format(country))
        country_input = []
        country_data = data[data[' Country'] == country]
        # While the end of the week is before the end of the specified end date, continue
        while datetime.datetime.strptime(en, date_format) < datetime.datetime.strptime(end_date, date_format):
            st, en = get_week_range(st)
            week_vector = get_week_vector(country_data, st, en)
            country_input.append(week_vector)
            st = pd.to_datetime(st) + pd.DateOffset(days=1)
            st = str(st).split()[0]
        input[country] = country_input
    return input
