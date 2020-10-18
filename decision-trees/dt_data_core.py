import warnings

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pydotplus
import graphviz
import os

if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    pd.set_option('display.max_rows', None)

    # Edit (09/30) cleaned_2.csv는 yes, no만 있습니다.
    # Edit (10/12) cleaned_2_again.csv는 불필요한 거 다 제거된 데이터입니다.

    os.listdir(os.getcwd())
    data = pd.read_csv("LLCP_cleaned_3.csv", decimal=',')

    data = pd.DataFrame(data)
    data = data.fillna(0).astype('float32')
    data = data.astype('int64')

    data_sp = data.sample(frac=0.05, random_state=3)

    # Edit (10/4): only include top 25 variables

    #X = np.array(data_sp.loc[:, data_sp.columns != "ADDEPEV2"])

    select_X = data_sp[['MENTHLTH', 'DECIDE', 'POORHLTH', 'HPLSTTST',
                          'EMPLOY1', 'SEX1', 'SMOKDAY2', 'DIFFDRES', 'HEIGHT3',
                          'AGE', 'DIFFALON', 'INCOME2', 'NUMHHOL3', 'RMVTETH4',
                          'SLEPTIM1', 'CHCCOPD1', 'SEATBELT', 'AVEDRNK2',
                          'PHYSHLTH']]

    X = np.array(select_X)

    Y = np.array(data_sp.loc[:, data_sp.columns == "ADDEPEV2"])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=3)

    clf = DecisionTreeClassifier(max_depth=4, random_state=3)
    clf = clf.fit(X_train, Y_train)

    pred = clf.predict(X_test)

    print("train score : {}".format(clf.score(X_train, Y_train)))
    #print(classification_report(Y_test, pred))
    print("test score : {}\n".format(clf.score(X_test, Y_test)))

    # Feature Importance

    '''

    print("Feature importance: \n{}".format(clf.feature_importances_))

    col_names = select_X.columns.tolist()

    print(pd.DataFrame({'col_name': clf.feature_importances_},
                       index=col_names).sort_values(by='col_name',
                                                          ascending=False))
    '''

    '''
    def plot_feature_importance_depress(model):
        n_features = X.shape[1]
        plt.barh(range(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), col_names)
        plt.xlabel("Feature importance")
        plt.ylabel("Features")
        plt.ylim(-1, n_features)
        plt.title("Feature importance of variables in the decision tree")
        plt.show()


    plot_feature_importance_depress(clf)


    # Visualization

    # Edit (10/4) : col_names.remove("ADDEPEV2")

    target_names = np.array(['Yes', 'No'])

    data_dot = tree.export_graphviz(clf, feature_names=col_names,
                                    class_names=target_names, filled=True,
                                    rounded=True, special_characters=True)

    dt_graph = pydotplus.graph_from_dot_data(data_dot)
    dt_graph.write_png("DT_core.png")
    '''

