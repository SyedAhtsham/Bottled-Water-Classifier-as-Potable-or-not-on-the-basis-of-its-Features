#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rc("font", size=14)
from sklearn.model_selection import train_test_split
import seaborn as sns
from imblearn.over_sampling import SMOTE

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv("datasetSheetFinal.csv", header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))

data.head()
print('\n')
data['Potability'].value_counts()

sns.countplot(x='Potability', data=data, palette='hls')
# plt.show()
plt.savefig('count_plot')


count_potables = len(data[data['Potability']==1])
count_non_potables = len(data[data['Potability']==0])
pct_of_non_potables = count_non_potables/(count_non_potables+count_potables)
print("percentage of non potables is ", pct_of_non_potables*100)
pct_of_potables = count_potables/(count_non_potables+count_potables)
print("percentage of potables is ", pct_of_potables*100)

data_final = data[["pH","EC","TDS","Ca","Mg","Hard","HCO3","Cl","Na","F","K","NO3","SO4","As","Potability"]]


X = data_final.loc[:, data_final.columns != 'Potability']
y = data_final.loc[:, data_final.columns == 'Potability']

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns


os_data_X,os_data_y = os.fit_resample(X_train, y_train)

os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
os_data_y= pd.DataFrame(data=os_data_y,columns=['Potability'])

X_train, X_test, y_train, y_test = train_test_split(os_data_X, os_data_y, test_size=0.3, random_state=0)
columns = X_train.columns

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of non potables in oversampled data",len(os_data_y[os_data_y['Potability']==0]))
print("Number of potables",len(os_data_y[os_data_y['Potability']==1]))
print("Proportion of non potables data in oversampled data is ",len(os_data_y[os_data_y['Potability']==0])/len(os_data_X))
print("Proportion of potables data in oversampled data is ",len(os_data_y[os_data_y['Potability']==1])/len(os_data_X))


min_max_scaler = preprocessing.MinMaxScaler()

X_train_minmax = min_max_scaler.fit_transform(X_train)

X_test_minimax = min_max_scaler.transform(X_test)


#
# logit_model=sm.Logit(y_train,X_train_minmax)
# result=logit_model.fit()
# print(result.summary2())

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train_minmax, y_train.values.ravel())

y_pred_prob = logreg.predict_proba(X_test_minimax)
y_pred = logreg.predict(X_test_minimax)

print("Accuracy of logistic regression classifier on training set: {:.2f}".format(logreg.score(X_train_minmax, y_train)*100))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test_minimax, y_test)*100))

type(y_pred_prob)

print("  NP Prob      P Prob   Class")
i = 0
for prob, result in zip(y_pred_prob, y_pred):
    if i==10:
        break
    print(str(prob) + " = " + str(result))
    i += 1


print()
print("Confusion Matrix:")
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)



