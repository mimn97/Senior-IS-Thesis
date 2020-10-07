from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image

import pandas as pd
import numpy as np

import pydotplus
import os
import warnings

if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')

    os.listdir(os.getcwd())
    tennis_data = pd.read_csv("play_tennis.csv")

    # Data Prepossessing

    tennis_data.outlook = tennis_data.outlook.replace('Sunny', 0)
    tennis_data.outlook = tennis_data.outlook.replace('Overcast', 1)
    tennis_data.outlook = tennis_data.outlook.replace('Rain', 2)

    tennis_data.temp = tennis_data.temp.replace('Hot', 3)
    tennis_data.temp = tennis_data.temp.replace('Mild', 4)
    tennis_data.temp = tennis_data.temp.replace('Cool', 5)

    tennis_data.humidity = tennis_data.humidity.replace('High', 6)
    tennis_data.humidity = tennis_data.humidity.replace('Normal', 7)

    tennis_data.wind = tennis_data.wind.replace('Weak', 8)
    tennis_data.wind = tennis_data.wind.replace('Strong', 9)

    tennis_data.play = tennis_data.play.replace('No', 10)
    tennis_data.play = tennis_data.play.replace('Yes', 11)

    # Data Separation for Training and Test Set

    X = np.array(pd.DataFrame(tennis_data,
                              columns=['outlook','temp', 'humidity', 'wind']))

    Y = np.array(pd.DataFrame(tennis_data, columns=['play']))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=1)

    # Training

    dt_clf = DecisionTreeClassifier()
    dt_clf = dt_clf.fit(X_train, Y_train)

    dt_prediction = dt_clf.predict(X_test)

    # Graph Visualization

    feature_names = tennis_data.columns.tolist()
    feature_names = feature_names[1:5]

    target_name = np.array(['Play No', 'Play Yes'])

    dt_dot_data = tree.export_graphviz(dt_clf, out_file=None,
                                       feature_names=feature_names,
                                       class_names=target_name,
                                       filled=True, rounded=True,
                                       special_characters=True)

    dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)
    dt_graph.write_png("dt_iris.png")
