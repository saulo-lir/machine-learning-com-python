import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


## Aplicando a separação para os dados da base 'credit_data.csv' ##
base = pd.read_csv('../data_bases/credit_data.csv')

# 1) Preencher os valores negativos com a média das idades de toda a base de dados, com excessão das idades negativas
media_idades = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = media_idades

# 2) Separando os atributos previsores da classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# 3) Substituir os valores faltantes pela média dos valores previsores
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# 4) Escalonar os valores previsores
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# 5) Separar os dados de treinamento dos dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0) # 25% da base de dados será usada para teste


## Aplicando a separação para os dados da base 'census.csv' ##
base = pd.read_csv('../data_bases/census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0) # 15% da base de dados será usada para teste