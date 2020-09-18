# Covid19-ML-Project
Machine Learning final project for the course Learning and Optimization 097209 @ Technion \
Predicting a classification based on number of cases per million each day 
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
We implemented BiLSTM and BiLSTM-CRF with PyTorch, we used both a manual implemented BiLSTM-CRF and also a version with pytorch-crf library. <br>
