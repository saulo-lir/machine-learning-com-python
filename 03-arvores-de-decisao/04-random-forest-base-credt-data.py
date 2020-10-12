import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

base = pd.read_csv('../data_bases/credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92

# 1) Separar os tipos das variáveis (Previsores e Classe)
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# 2) Aplicar o pré processamento

# Substituir os valores faltantes pela média dos valores previsores
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# Escalonamento
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Separar os dados de treinamento dos dados de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# 3) Aplicar o treinamento
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0) # n_estimators = Número de árvores
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# 4) Comparar os valores previstos com os da base de teste
precisao = accuracy_score(classe_teste, previsoes)
print(precisao)
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)