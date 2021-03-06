import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer

###### Pre processamento dos dados #######

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

###### Treinamento e previsão dos dados - Criação do classificador #######

classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

print(previsoes)
print(classe_teste)

# Percentual de acertos (acurácia)
precisao = accuracy_score(classe_teste, previsoes)
print(precisao)

# Matriz de confusão, contendo os registros onde ocorrem os erros e acertos
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)