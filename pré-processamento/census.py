import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

census = pd.read_csv('../data/census.csv')

scaler_census = StandardScaler()


# Estatisticas iniciais
# print(census.describe())
# print(census.isnull().sum())

# print(np.unique(census['income'], return_counts= True))

# grafico barra simples
# sns.countplot(x = census['income'])

# plt.hist(x = census['age'])
# plt.hist(x = census['education-num'])
# plt.hist(x = census['hour-per-week'])

# graph = px.treemap(census, path=['workclass', 'age'])
# graph = px.treemap(census, path=['occupation', 'relationship', 'age'])
# graph = px.parallel_categories(census, dimensions=['occupation', 'relationship'])
# graph = px.parallel_categories(census, dimensions=['workclass','occupation', 'income'])

x_census = census.iloc[:, 0:14].values
y_census = census.iloc[:, 14].values

# x_census = scaler_credit.fit_transform(x_census)

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:,1] = label_encoder_workclass.fit_transform(x_census[:,1])
x_census[:,3] = label_encoder_education.fit_transform(x_census[:,3])
x_census[:,5] = label_encoder_marital.fit_transform(x_census[:,5])
x_census[:,6] = label_encoder_occupation.fit_transform(x_census[:,6])
x_census[:,7] = label_encoder_relationship.fit_transform(x_census[:,7])
x_census[:,8] = label_encoder_race.fit_transform(x_census[:,8])
x_census[:,9] = label_encoder_sex.fit_transform(x_census[:,9])
x_census[:,13] = label_encoder_country.fit_transform(x_census[:,13])

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
x_census = onehotencoder_census.fit_transform(x_census).toarray()

x_census = scaler_census.fit_transform(x_census)

x_census_train, x_census_test, y_census_train, y_census_test = train_test_split(x_census, y_census, test_size=0.15, random_state=0)

with open('../data/census.pkl', mode='wb') as f:
    pickle.dump([x_census_train, x_census_test, y_census_train, y_census_test], file=f)

# graph.show()
# plt.show()

