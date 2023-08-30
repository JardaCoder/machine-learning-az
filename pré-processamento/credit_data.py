import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

base_credit = pd.read_csv('../data/credit_data.csv')
scaler_credit = StandardScaler()

# print(base_credit.describe())
# print(base_credit[base_credit['income'] >= 69995.685578])
# print(base_credit[base_credit['loan'] <= 1.377630])

# conta quantos empréstimos quitados ou atrasados
# sns.countplot(x = base_credit['default'])

# histograma por idades
# plt.hist(x = base_credit['age'])

# histograma por renda
# plt.hist(x = base_credit['income'])

# histograma por divida
# plt.hist(x = base_credit['loan'])

# Cria graficos cruzando as informações
# graph = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')

#  --------------- --------------- tratar valores inconsistentes

#  localizar registros
# print(base_credit.loc[base_credit['age'] < 0])

# Apagar a coluna inteira de todos os registros zoados.
# base_credit2 = base_credit.drop('age', axis= 1)

# Apaga registros errados
# base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
# print(base_credit3)

# Preencher com dados existentes de verdade ( melhor metodo)

# tirar média dos valores
# print(base_credit.mean())
# print(base_credit['age'][base_credit['age'] > 0].mean())

base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)

assert base_credit.loc[base_credit['age'] < 0, 'age'].empty

# print(base_credit.head(27))

#  --------------- --------------- tratar valores inconsistentes

#  --------------- --------------- tratar valores faltantes

# conta quantos nulos por coluna
# print(base_credit.isnull().sum())

# print(base_credit.loc[base_credit['age'].isnull()])

# fill
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)
assert base_credit['age'].loc[base_credit['age'].isnull()].empty

#  --------------- --------------- tratar valores faltantes

#  --------------- --------------- previoses e classe
x_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4].values

# Aplica o escalonamento com o metodo padronização.
x_credit = scaler_credit.fit_transform(x_credit)

x_credit_train, x_credit_test, y_credit_train, y_credit_test = train_test_split(x_credit, y_credit, test_size=0.25, random_state=0)

with open('../data/credit.pkl', mode='wb') as f:
    pickle.dump([x_credit_train, y_credit_train, x_credit_test, y_credit_test], file=f)

# graph.show()
# plt.show()
