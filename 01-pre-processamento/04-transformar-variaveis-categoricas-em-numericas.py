import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

base = pd.read_csv('../data_bases/census.csv')
print(base)

# Transformar variáveis categóricas em numéricas. O objetivo é facilitar o processamento dos algoritmos de machine learning

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Converter os valores das colunas categóricas para numéricos
labelencoder_previsores = LabelEncoder()
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13])
print(previsores)

# Para melhorar a transformação das variáveis, às vezes é necessário aplicar a técnica das variáveis 'Dummy'

onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

print(previsores)

# Aplicando a transformação categórica->numérica nas variáveis classe
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

print(classe)

# Aplicando o escalonamento em todas as variáveis, indeoendente se elas são do tipo Dummy ou não
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
print(previsores)