import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

###### Pre processamento dos dados #######

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

# 4) Escalonar os valores previsores (Colocar os valores numéricos na mesma escala)
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# 5) Separar os dados de treinamento dos dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0) # 25% da base de dados será usada para teste


###### Treinamento e previsão dos dados - Criação do classificador #######

# 1) Aplicar o treinamento

classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

# Se os previsores estiverem iguais a classe_teste, então o algoritmo foi bem treinado
# print(classe_teste)
# print(previsoes)

# 2) Comparar os valores previstos com os da base de teste

# Percentual de acertos (acurácia)
precisao = accuracy_score(classe_teste, previsoes)
print(precisao)

# Matriz de confusão, contendo os registros onde ocorrem os erros e acertos
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)