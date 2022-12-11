#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from keras.applications.densenet import layers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python import tf2
from keras.models import Sequential
from keras.layers import Dense

# file = open('datasetSheetFinal1.csv', "w")

df = pd.read_csv("datasetSheetFinal1.csv")
# df.drop(df.columns[[0,1]], axis=1, inplace=True)
#
# df1 = df[df['Potability']==0];
# df2 = df[df['Potability']==1][0:222];


# arrNonPot = df1.values
# arrPot = df2.values

# frames = [df1,df2]
#
# finalDF = pd.concat(frames)


# file.write("pH,EC,TDS,Ca,Mg,Hard,HCO3,Cl,Na,F,K,NO3,SO4,As,Potability"+'\n')

# for item1, item2 in zip(arrPot, arrNonPot):
#     for i in range(len(item1)):
#         file.write(str(item1[i]))
#         if i==len(item1)-1:
#             continue
#         file.write(",")
#     file.write("\n")
#     for i in range(len(item2)):
#         file.write(str(item2[i]))
#         if i==len(item2)-1:
#             continue
#         file.write(",")
#     file.write("\n")



# df = pd.read_csv('datasetSheetFinal1.csv')
# file = open("report.csv", "w")
#
# file.write("HL(s),HL1_N,HL2_N,HL3_N,HL4_N,HL5_N,HL1_AF,HL2_AF,HL3_AF,HL4_AF,HL5_AF,OL_AF,Accuracy")



#
# count_potables = len(finalDF[finalDF['Potability']==1])
# count_non_potables = len(finalDF[finalDF['Potability']==0])
# print(count_non_potables)
# print(count_potables)
#
# finalDF = finalDF.sample(frac=1).reset_index(drop=True)




# print(finalDF['Potability'])

dataset = df.values

X = dataset[:,0:14]
Y = dataset[:,14]





min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)


X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

# print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
print(X_train.shape, X_val_and_test.shape, Y_train.shape, Y_val_and_test.shape)

early_stopping_monitor = EarlyStopping(patience=10)

sums = 0
sums2 = 0

# for i in range(4):
model = Sequential()
model = Sequential([
Dense(15, activation="relu", input_shape=(14,)),

Dense(15, activation="relu"),
Dense(15, activation="relu"),
Dense(15, activation="relu"),
Dense(15, activation="relu"),
Dense(15, activation="relu"),
Dense(15, activation="relu"),



        Dense(1, activation="sigmoid"),
        # Dense(32, activation="relu"),
        #
        # Dense(1, activation="sigmoid"),


])

model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, validation_data=(X_val, Y_val),
              epochs=100, callbacks=[early_stopping_monitor])
accuracy1 = model.evaluate(X_test, Y_test)[1]
# sums += accuracy
accuracy2 = model.evaluate(X_train, Y_train)[1]
# sums2 += accuracy
keras.backend.clear_session()
print()
print()

print("Accuracy on Training: " + str(round(accuracy2, 3)*100) + "%")
print()
print("Accuracy on Testing: " + str(round(accuracy1, 3)*100) + "%")

# no_of_neurons = [32,64,128, 256]
#
#
# activation_functions = ["sigmoid","relu","tanh","softmax"]
#
# no_of_layers = [32,64,128]

# for n in range(3):  #no of hidden layers
#     for i in range(4):  #activation function loop
#         file.write("\n")
#
#

# i = 0
# for j in range(4):  # no of neurons in each layer
#     file.write("\n")
#     model = keras.Sequential()
#     model.add(
#         layers.Dense(no_of_neurons[j], activation=activation_functions[i], kernel_initializer='random_uniform',
#                      bias_initializer='zeros', input_shape=(14,)))
#     for k in range(50):
#         model.add(layers.Dense(no_of_neurons[j], activation=activation_functions[i]))
#
#     model.add(layers.Dense(2, activation=activation_functions[i]))
#
#     # model = Sequential([
#     #     Dense(no_of_neurons[j], activation=activation_functions[i], input_shape=(14,)),
#     #     Dense(2, activation=activation_functions[i]),
#     # ])
#     #
#
#     model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
#
#     hist = model.fit(X_train, Y_train, batch_size=32, validation_split=0.2, epochs=50,
#                      callbacks=[early_stopping_monitor])
#     print()
#     print()
#     accuracy = model.evaluate(X_val_and_test, Y_val_and_test)[1]
#     file.write(str(30) + "," + str(no_of_neurons[j]) + "," + activation_functions[
#         i] + "," + "" + "," + "" + "," + "" + "," + "" + "," + "" + "," + str(round(accuracy, 3)))
#
#     keras.backend.clear_session()
#     i += 1

