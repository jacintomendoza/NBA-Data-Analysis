import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# R E A D I N G    D A T A
data = pd.read_csv('NBAstats.csv')

# U S E F U L    A T T R I B U T E S
feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
'3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
#################################################
#Pandas DataFrame allows you to select columns.
#We use column selection to split the data into features and class.

# C L E A N S    I N C O N C L U S I V E    V A L U E S
data = data[data.G > 25]

# D E F I N E S    H I G H    3  -  P O I N T E R S
data['high_3p%'] = (data['3P%'] > .30) *1

# y - T A R G E T    V A R I A B L E
y = data['Pos'].copy()
y.head()
# print(y)

# x, F E A T U R E D    D A T A, independent features
x = data[feature_columns].copy()
x.columns
# print(x)

# M O D E L    T R A I N I N G                            75% in train and 25% in test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .25, random_state = 324)

three_perc_classifier = DecisionTreeClassifier(max_leaf_nodes = 15, random_state = 0)
three_perc_classifier.fit(X_train, y_train)

# P R E D I C T    R E S P O N S E    F O R    T E S T    D A T A S E T
y_predicted = three_perc_classifier.predict(X_test)

# S C O R E S  /  A C C U R A C Y
print("\nAccuracy:", accuracy_score(y_test, y_predicted))
print("Training set score: {:.3f}".format(three_perc_classifier.score(X_train, y_train)))
print("Test set score: {:.3f}".format(three_perc_classifier.score(X_test, y_test)))

#print('Accuracy Score: ' + str(accuracy_score(y_test, y_predicted)) + ' %') # Already have

# C O N F U S I O N    M A T R I X
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predicted))

# C R O S S - V A L I D A T I O N
tree = DecisionTreeClassifier(max_depth = 4, random_state = 0)
scores = cross_val_score(tree, x, y, cv = 10)
print("\nCross-validation scores: {}".format(scores))
print("\nAverage cross-validation score: {:.2f}".format(scores.mean()))

# S O U R C E S
# https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
# https://www.youtube.com/watch?v=zvFot5vs6aQ
