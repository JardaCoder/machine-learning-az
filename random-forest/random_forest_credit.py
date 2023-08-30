import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from yellowbrick.classifier import ConfusionMatrix

from utils import get_classes_name

with open('../data/credit.pkl', 'rb') as f:
    x_credit_train, y_credit_train, x_credit_test, y_credit_test = pickle.load(f)

credit_risk_tree = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
credit_risk_tree.fit(x_credit_train, y_credit_train)

previsions = credit_risk_tree.predict(x_credit_test)

print(accuracy_score(y_credit_test, previsions))
# 98.4%
cm = ConfusionMatrix(credit_risk_tree)
cm.fit(x_credit_train, y_credit_train)
cm.score(x_credit_test, y_credit_test)
cm.show()
print(classification_report(y_credit_test, previsions))

#
# forecasters = ['income', 'age', 'loan']
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
#
# tree.plot_tree(credit_risk_tree, feature_names=forecasters, class_names=get_classes_name(credit_risk_tree.classes_),  filled=True)
# fig.savefig('arvore_credit.png')