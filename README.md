# Covid19-ML-Project
Machine Learning final project for the course Learning and Optimization 097209 @ Technion \
Predicting a classification based on number of COVID-19 cases per million each day 
### Classes used
```
# 0 - no new cases on this day
# 1 - 1-10 new cases per million a day
# 2 - 11-100 new cases per million a day
# 3 - 101-250 new cases per million a day
# 4 - more than 250 per million new cases a day
```
### Data
For this project we used joined data from https://ourworldindata.org/coronavirus

### Classifiers
This part holds the models we used to evaluate our performace. <br>
We implemented Logistic Regression, SVM and Random Forest with sklearn. <br>
We implemented BiLSTM and BiLSTM-CRF with PyTorch, we used both a manual implemented BiLSTM-CRF and also a version with pytorch-crf library with variations of different sequence lengths. <br>

### SEIR Model
This part holds the code for the creative part, for this part we used NetworkX libraby to generate graphs and implemented our simulations.

### Utilities
This part holds scripts that were implemented by us to preprocess the raw data we collected and convert it to the formats used by our models, we used mainly Pandas library for this part.
