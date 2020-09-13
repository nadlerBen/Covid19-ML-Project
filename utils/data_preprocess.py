import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import random


# %%
covid = pd.read_excel('data_project_almost.xlsx')
covid = covid.drop([' Country_code', 'merged', ' WHO_region'], axis=1)

# classes:
# 0 - no new cases this day
# 1 - 1-10 new cases per million today
# 2 - 11-100 new cases per million today
# 3 - 101-250 new cases per million this day
# 4 - more than 250 per million new cases this day


classes = []
for idx, cases_per_m in enumerate(covid['new cases per millions']):
    # idx 263 and 1553 contains negative numbers - check why
    if cases_per_m <= 0:
        classes.append(0)
    elif 0 < cases_per_m <= 10:
        classes.append(1)
    elif 10 < cases_per_m <= 100:
        classes.append(2)
    elif 100 < cases_per_m <= 250:
        classes.append(3)
    elif cases_per_m > 250:
        classes.append(4)


covid['target'] = classes

# removed numeric target column
covid = covid.drop(['new cases per millions'], axis=1)
# filling missin data with MICE method for missing data
imp = IterativeImputer(max_iter=10, random_state=0)
covid_imp = covid.drop(['Date_reported', ' Country'], axis=1)
imp.fit(covid_imp.to_numpy())
filled = imp.transform(covid_imp.to_numpy())
for idx, column in enumerate(covid.columns[2:]):
    covid[column] = filled[:, idx]
covid.to_csv('project_data_preprocessed.csv')
covid = pd.read_csv('project_data_preprocessed.csv')
for target in set(covid['target']):
    print(target, len(covid[covid['target'] == target]))
