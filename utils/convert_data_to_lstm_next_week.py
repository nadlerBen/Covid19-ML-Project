import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler


def load_data():
    covid = pd.read_csv('project_data_preprocessed.csv')
    covid['Date'] = pd.to_datetime(covid['Date_reported'])
    covid = covid.set_index(covid['Date'])
    covid = covid.drop([' New_cases', ' Cumulative_cases',
                        'Unnamed: 0', 'Date_reported'], axis=1)
    return covid


def get_week_range(start_date: str) -> str:
    end_date = pd.to_datetime(start_date) + pd.DateOffset(days=6)
    end_date = str(end_date).split()[0]
    return start_date, end_date


def get_week_vector(data, current_start_date: str, current_end_date: str, next_start_date: str, next_end_date: str):
    scaler = MinMaxScaler()
    current_week_data = data.loc[current_start_date:current_end_date]
    next_week_data = data.loc[next_start_date:next_end_date]
    current_week_data = current_week_data.drop([' Country', 'Date'], axis=1)
    current_week_data[current_week_data.columns[:-1]
                      ] = scaler.fit_transform(current_week_data[current_week_data.columns[:-1]])
    next_week_data = next_week_data.drop([' Country', 'Date'], axis=1)
    next_week_data[next_week_data.columns[:-1]
                   ] = scaler.fit_transform(next_week_data[next_week_data.columns[:-1]])
    features = current_week_data[current_week_data.columns[:-1]
                                 ].to_numpy().tolist()
    classes = next_week_data[next_week_data.columns[-1]].to_numpy().tolist()
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
        country_input = []
        country_data = data[data[' Country'] == country]
        # While the end of the week is before the end of the specified end date, continue
        while datetime.datetime.strptime(en, date_format) < datetime.datetime.strptime(end_date, date_format):
            st, en = get_week_range(st)
            next_week_st = pd.to_datetime(en) + pd.DateOffset(days=1)
            next_week_st, next_week_en = get_week_range(next_week_st)
            week_vector = get_week_vector(
                country_data, st, en, next_week_st, next_week_en)
            country_input.append(week_vector)
            st = pd.to_datetime(st) + pd.DateOffset(days=1)
            st = str(st).split()[0]
        input[country] = country_input
    return input
