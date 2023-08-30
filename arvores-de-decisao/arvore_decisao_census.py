from sklearn.metrics import accuracy_score, classification_report
import pickle

from sklearn.tree import DecisionTreeClassifier

with open('../data/census.pkl', 'rb') as f:
    x_census_train, x_census_test, y_census_train, y_census_test = pickle.load(f)

census_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
census_tree.fit(x_census_train, y_census_train)

previsions = census_tree.predict(x_census_test)

print(accuracy_score(y_census_test, previsions))

# cm = ConfusionMatrix(census_tree)
# cm.fit(x_census_train, y_census_train)
# cm.score(x_census_test, y_census_test)
# cm.show()
# 81%
print(classification_report(y_census_test, previsions))
