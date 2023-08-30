import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix

from utils import get_classes_name

with open('../data/census.pkl', 'rb') as f:
    x_census_train, x_census_test, y_census_train, y_census_test = pickle.load(f)

census_tree = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
census_tree.fit(x_census_train, y_census_train)

previsions = census_tree.predict(x_census_test)

print(accuracy_score(y_census_test, previsions))

cm = ConfusionMatrix(census_tree)
cm.fit(x_census_train, y_census_train)
cm.score(x_census_test, y_census_test)
cm.show()

print(classification_report(y_census_test, previsions))
