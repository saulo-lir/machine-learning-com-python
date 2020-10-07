import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Ler arquivo .csv
base = pd.read_csv('../data_bases/credit_data.csv') # A vairável base será do tipo DataFrame

# Exibir dados estatísticos da planilha
print(base.describe())

# 2) Tratar dados faltantes
# Localizar onde existe valores null
print(base.loc[pd.isnull(base['age'])])


'''
Armazenar os atributos previsores da planilha

: = Seleciona toda a linha
1:4 = Seleciona do atributo da linha 1 até a linha 3 (setamos o 4 mas ele não é considerado)
No caso, os atributos previsores são: income, age, loan (Não é interessante pegarmos os ids dos itens, pois não oferecem nenhuma informação relevante para o algoritmo)
'''
previsores = base.iloc[:, 1:4].values
print(previsores)

'''
Selecionar o atributo classe
'''

classe = base.iloc[:, 4].values
print(classe)


# Substituir os valores faltantes pela média dos valores previsores
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])
print(previsores[:, 0:3])