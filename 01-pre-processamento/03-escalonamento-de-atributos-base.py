import pandas as pd
from sklearn.preprocessing import StandardScaler

# O objetivo do escalonamento é aproximar os valores previsores, utilizando uma escala de aproximação. Isso ajuda o algoritmo a não tratar com mais importância os valores maiores, além de melhorar seu desempenho.

base = pd.read_csv('../data_bases/credit_data.csv')

previsores = base.iloc[:, 1:4].values
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

print(previsores)