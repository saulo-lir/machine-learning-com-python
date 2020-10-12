import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export

base = pd.read_csv('../data_bases/risco_credito.csv')

# 1) Separar os tipos das variáveis (Previsores e Classe)
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# 2) Aplicar o pré processamento dos valores para converter os atributos categóricos em discretos. Não é possível gerar a árvore de decisão a partir de atributos categóricos.
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
#print(previsores)

# 3) Aplicar o treinamento
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe)
print(classificador.feature_importances_) # Exibir a importância de cada atributo
print(classificador.score(previsores, classe)) # Verificar se o modelo está bom. Quanto mais próximo de 1 (100%) melhor.

# 4) Exportar o gráfico de árvore de decisão para ser lido posteriormente pela ferramenta graphviz
export.export_graphviz(classificador,
                       out_file = 'arvore.dot', # .dot é a extensão utilizada pela ferramenta graphviz
                       feature_names = ['historia', 'divida', 'garantias', 'renda'],
                       class_names = classificador.classes_,
                       filled = True,
                       leaves_parallel=True)

# 5) Aplicar uma classificação com modelo já treinado, usando os seguintes exemplos:
'''
história boa, dívida alta, garantias nenhuma, renda > 35
história ruim, dívida alta, garantias adequada, renda < 15
'''

resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
print(resultado)