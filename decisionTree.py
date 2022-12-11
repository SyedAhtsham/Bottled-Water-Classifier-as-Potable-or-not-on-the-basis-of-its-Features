#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import graphviz
import numpy
import pandas as pd
import numpy as np
from numpy import *
from pandas import DataFrame
from sklearn.tree import export_graphviz
from sklearn import *
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.model_selection import train_test_split
import seaborn as sns
from imblearn.over_sampling import SMOTE



def compute_entropy(S):
    values = S['Potability'].value_counts()
    size = S.count()
    entropy = -(values[0]/size) * log2(values[0]/size)-(values[1]/size) * log2(values[1]/size)
    return float(round(entropy, 3))

def compute_feature_entropy(feature):

    probs = feature.value_counts(normalize=True)

    entropy = -1 * np.sum(np.log2(probs) * probs)

    return round(entropy, 3)

def comp_feature_gain_ratio(info_gain, weight_list):
    w_l = np.array(weight_list)
    w_l = np.log2(w_l)
    feature_split_info = -1 * np.sum(np.array(weight_list) * w_l)
    gain_ratio = info_gain / feature_split_info

    return gain_ratio


def comp_feature_information_gain(df, target, descriptive_feature):


    target_entropy = compute_feature_entropy(df[target])

    # entropy_list to store the entropy of each partition
    # weight_list to store the relative number of observations in each partition
    entropy_list = list()
    weight_list = list()

    # loop over each level of the descriptive feature
    # to partition the dataset with respect to that level
    # and compute the entropy and the weight of the level's partition
    for level in df[descriptive_feature].unique():
        df_feature_level = df[df[descriptive_feature] == level]
        entropy_level = compute_feature_entropy(df_feature_level[target])
        entropy_list.append(round(entropy_level, 3))
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(weight_level)

    feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))
    information_gain = target_entropy - feature_remaining_impurity
    gain_ratio = comp_feature_gain_ratio(information_gain, weight_list)
    print('Feature: ', descriptive_feature)
    print('information gain: {:.3f}'.format(information_gain))
    print('gain ratio: {:.3f}'.format(gain_ratio))

    print('====================')



    return information_gain, gain_ratio





def main():
    sns.set(style="white")
    sns.set(style="whitegrid", color_codes=True)

    data = pd.read_csv("datasetSheetFinal.csv", header=0)
    data = data.dropna()


    data.head()
    print('\n')
    data['Potability'].value_counts()

    sns.countplot(x='Potability', data=data, palette='hls')
    # plt.show()
    plt.savefig('count_plot')

    count_potables = len(data[data['Potability'] == 1])
    count_non_potables = len(data[data['Potability'] == 0])
    pct_of_non_potables = count_non_potables / (count_non_potables + count_potables)
    # print("percentage of non potables is ", pct_of_non_potables * 100)
    pct_of_potables = count_potables / (count_non_potables + count_potables)
    # print("percentage of potables is ", pct_of_potables * 100)

    data_final = data[
        ["pH", "EC", "TDS", "Ca", "Mg", "Hard", "HCO3", "Cl", "Na", "F", "K", "NO3", "SO4", "As", "Potability"]]

    X = data_final.loc[:, data_final.columns != 'Potability']
    y = data_final.loc[:, data_final.columns == 'Potability']

    os = SMOTE(random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    columns = X_train.columns

    os_data_X, os_data_y = os.fit_resample(X_train, y_train)

    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['Potability'])

    # splitting training and test set into 70:30
    X_train, X_test, y_train, y_test = train_test_split(os_data_X, os_data_y, test_size=0.3, random_state=0)
    columns = X_train.columns

    df = DataFrame.copy(X_train)

    df.insert(14,"Potability", y_train['Potability'], True)



    # we can Check the numbers of our data
    # print("length of oversampled data is ",len(os_data_X))
    # print("Number of non potables in oversampled data",len(os_data_y[os_data_y['Potability']==0]))
    # print("Number of potables",len(os_data_y[os_data_y['Potability']==1]))
    # print("Proportion of non potables data in oversampled data is ",len(os_data_y[os_data_y['Potability']==0])/len(os_data_X))
    # print("Proportion of potables data in oversampled data is ",len(os_data_y[os_data_y['Potability']==1])/len(os_data_X))
    #

    X_train['pH'] = pd.cut(x=X_train.pH, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_train['EC'] = pd.cut(x=X_train.EC, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_train['TDS'] = pd.cut(x=X_train.TDS, bins=3, labels=None, retbins=False, duplicates='raise',
                            ordered=True).cat.codes
    X_train['Ca'] = pd.cut(x=X_train.Ca, bins=4, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_train['Mg'] = pd.cut(x=X_train.Mg, bins=5, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_train['HCO3'] = pd.cut(x=X_train.HCO3, bins=4, labels=None, retbins=False, duplicates='raise',
                             ordered=True).cat.codes
    X_train['Cl'] = pd.cut(x=X_train.Cl, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_train['Na'] = pd.cut(x=X_train.Na, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_train['F'] = pd.cut(x=X_train.F, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_train['K'] = pd.cut(x=X_train.K, bins=4, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_train['NO3'] = pd.cut(x=X_train.NO3, bins=4, labels=None, retbins=False, duplicates='raise',
                            ordered=True).cat.codes
    X_train['SO4'] = pd.cut(x=X_train.SO4, bins=3, labels=None, retbins=False, duplicates='raise',
                            ordered=True).cat.codes
    X_train['As'] = pd.cut(x=X_train.As, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes

    #######################
    #######################
    #######################

    X_test['pH'] = pd.cut(x=X_test.pH, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_test['EC'] = pd.cut(x=X_test.EC, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_test['TDS'] = pd.cut(x=X_test.TDS, bins=3, labels=None, retbins=False, duplicates='raise',
                            ordered=True).cat.codes
    X_test['Ca'] = pd.cut(x=X_test.Ca, bins=4, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_test['Mg'] = pd.cut(x=X_test.Mg, bins=5, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_test['HCO3'] = pd.cut(x=X_test.HCO3, bins=4, labels=None, retbins=False, duplicates='raise',
                             ordered=True).cat.codes
    X_test['Cl'] = pd.cut(x=X_test.Cl, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_test['Na'] = pd.cut(x=X_test.Na, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_test['F'] = pd.cut(x=X_test.F, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_test['K'] = pd.cut(x=X_test.K, bins=4, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes
    X_test['NO3'] = pd.cut(x=X_test.NO3, bins=4, labels=None, retbins=False, duplicates='raise',
                            ordered=True).cat.codes
    X_test['SO4'] = pd.cut(x=X_test.SO4, bins=3, labels=None, retbins=False, duplicates='raise',
                            ordered=True).cat.codes
    X_test['As'] = pd.cut(x=X_test.As, bins=3, labels=None, retbins=False, duplicates='raise', ordered=True).cat.codes



    tree1 = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree1.fit(X_train, y_train)
    tree.plot_tree(tree1)
    y_pred = tree1.predict(X_test)

    export_graphviz(tree1, out_file="tree.dot", class_names=["Potable", "Non-Potable"],
                    feature_names=columns, impurity=False, filled=True)

    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)

    tree2 = tree.DecisionTreeClassifier(criterion='log_loss', random_state=0)
    tree2.fit(X_train, y_train)
    tree.plot_tree(tree2)

    export_graphviz(tree2, out_file="tree2.dot", class_names=["Potable", "Non-Potable"],
                    feature_names=columns, impurity=False, filled=True)

    with open("tree2.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)

    print("Accuracy on training set using Information Gain: {:.3f}".format(tree1.score(X_train, y_train)*100))
    print("Accuracy on test set Information Gain: {:.3f}".format(tree1.score(X_test, y_test)*100))

    print("Accuracy on training set using Gain Ratio: {:.3f}".format(tree2.score(X_train, y_train)*100))
    print("Accuracy on test set using Gain Ratio: {:.3f}".format(tree2.score(X_test, y_test)*100))

    print()


    max_info = 0.00000000000001
    max_ratio = 0.00000000000001
    top_feature_info = ''
    top_feature_ratio = ''
    for feature in df.drop(columns='Potability').columns:
        feature_info_gain, feature_gain_ratio = comp_feature_information_gain(df, 'Potability', feature)

        if max_info < feature_info_gain:
            max_info = feature_info_gain
            top_feature_info = feature
        if max_ratio < feature_gain_ratio:
            top_feature_ratio = feature
            max_ratio = feature_gain_ratio




    print("Top Selected Feature using Information Gain: "+top_feature_info + " with Info_Gain: {:.4f}".format(max_info))
    print("Top Selected Feature using Gain Ratio: " + top_feature_ratio + " with Gain_Ratio: {:.4f}".format(max_ratio))

    print()
    print("Confusion Matrix:")
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

if __name__ == "__main__":
    main()

