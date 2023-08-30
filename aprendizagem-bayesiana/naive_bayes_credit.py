from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
import pickle
from yellowbrick.classifier import ConfusionMatrix

# Aplicações do algoritmo
# Probabilistica
# Filtros de spam, Mineração de emoções, Separação de Documentos
# Se não executar com o escalonamento parece melhor


with open('../data/credit.pkl', 'rb') as f:
    x_credit_train, y_credit_train, x_credit_test, y_credit_test = pickle.load(f)

    naive_credit_data = GaussianNB()
    naive_credit_data.fit(x_credit_train, y_credit_train)

    previsions = naive_credit_data.predict(x_credit_test)

    accuracy_score(y_credit_test, previsions)

    cm = ConfusionMatrix(naive_credit_data)
    cm.fit(x_credit_train, y_credit_train)
    cm.score(x_credit_test, y_credit_test)

    # cm.show()

    print(classification_report(y_credit_test, previsions))