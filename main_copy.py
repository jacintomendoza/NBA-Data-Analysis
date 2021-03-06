import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz

# R E A D I N G    D A T A
nba = pd.read_csv('NBAstats.csv')

# print the column names
original_headers = list(nba.columns.values)
#print(original_headers)

#print the first three rows.
#print(nba[0:3])

# "Position (pos)" is the class attribute we are predicting.
class_column = 'Pos'
#The dataset contains attributes such as player name and team name.
#We know that they are not useful for classification and thus do not
#include them as features.
feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
'3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
#################################################
#Pandas DataFrame allows you to select columns.
#We use column selection to split the data into features and class.
nba_feature = nba[feature_columns]
nba_class = nba[class_column]
# print(nba_feature[0:3]) # Lists val of Age, G, GS.....
# print(list(nba_class[0:3])) # PF, SG, C

train_feature, test_feature, train_class, test_class = \
train_test_split(nba_feature, nba_class, stratify=nba_class, \
train_size=0.75, test_size=0.25)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_feature, train_class)
    # record training set accuracy
    training_accuracy.append(knn.score(train_feature, train_class))
    # record generalization accuracy
    test_accuracy.append(knn.score(test_feature, test_class))


knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=1)
knn.fit(train_feature, train_class)
prediction = knn.predict(test_feature)
print("Test set predictions:\n{}".format(prediction))
print("Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))

train_class_df = pd.DataFrame(train_class,columns=[class_column])
train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)

train_data_df.to_csv('train_data.csv', index=False)

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()
