import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

'''
GaussianNB é a classe responsável pelo algoritmo de treinamento Naive Bayes.

O método fit() da classe GaussianNB não aceita atributos categóricos (strings), então é necessário fazermos o pré processamento dos dados para convertê-los em discretos.
'''

# 1) Importar a base de dados
base = pd.read_csv('../data_bases/risco_credito.csv')

# 2) Separar os tipos das variáveis (Previsores e Classe)
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# 3) Aplicar o pré processamento dos valores para converter os atributos categóricos em discretos
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
#print(previsores)

# 4) Aplicar o método da classe responsável pelo treinamento
classificador = GaussianNB()
classificador.fit(previsores, classe) # Gerar a tabela de probabilidade dos valores

# 5) Aplicar a previsão dos dados conforme é adicionado um novo registro no algoritmo treinado

'''
história boa, dívida alta, garantias nenhuma, renda > 35 = [0,0,1,2]
história ruim, dívida alta, garantias adequada, renda < 15 = [3, 0, 0, 0]
'''

resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(resultado)

# 6) Visualizar algumas propriedades do classificador
print(classificador.classes_) # Exibir todas as classes identificadas
print(classificador.class_count_) # Exibir a contagem de cada classe
print(classificador.class_prior_) # Exibir as probabilidades apriori. Quais as probabilidades de ocorrência de cada classe.
