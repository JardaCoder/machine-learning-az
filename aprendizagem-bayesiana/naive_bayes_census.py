from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
import pickle
from yellowbrick.classifier import ConfusionMatrix

# Aplicações do algoritmo
# Probabilistica
# Filtros de spam, Mineração de emoções, Separação de Documentos


with open('../data/census.pkl', 'rb') as f:
    x_census_train, x_census_test, y_census_train, y_census_test = pickle.load(f)

    naive_census_data = GaussianNB()
    naive_census_data.fit(x_census_train, y_census_train)

    previsions = naive_census_data.predict(x_census_test)

    accuracy_score(y_census_test, previsions)

    cm = ConfusionMatrix(naive_census_data)
    cm.fit(x_census_train, y_census_train)
    cm.score(x_census_test, y_census_test)

    # cm.show()

    print(classification_report(y_census_test, previsions))
