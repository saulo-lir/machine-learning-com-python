import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from yellowbrick.regressor import ResidualsPlot

base = pd.read_csv('../data_bases/plano_saude.csv')

# 1) Separar as variáveis X e y, onde X são as variáveis independentes, e y as dependentes, ou seja, que queremos fazer a previsão

X = base.iloc[:, 0].values
y = base.iloc[:, 1].values
print(X)
print(y)

# 2) Verificar a correlação entre as variáveis independentes com as dependentes (o quanto uma variável está próxima de outra)

correlacao = np.corrcoef(X, y) # Caluculando o coeficiente de correlação
print(correlacao)

# 3) Converter os valores de X em Matriz para poder ser utilizada pela bilbioteca do sklearn

X = X.reshape(-1,1) # -1 = Desconsiderar as linhas, 1 = adicionar uma coluna
print(X)

# 4) Treinar o modelo de regressão linear

regressor = LinearRegression()
regressor.fit(X, y)

# 5) Visualizar os parâmetros da fórmula matemática de regressão linear

# b0
b0 = regressor.intercept_
print(b0)

#b1
b1 = regressor.coef_
print(b1)

# 5) Plotar o gráfico de regressão linear

plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Custo')
plt.show()

# 6) Após treinar o modelo, podemos aplicar algumas previsões

# Prever o preço do plano para uma pessoa com 40 anos de idade (Podemos utilizar os dois formatos abaixo para obter o mesmo resultado)
previsao1 = regressor.intercept_ + regressor.coef_ * 40
previsao2 = regressor.predict(np.array(40).reshape(1, -1))
print(previsao1)
print(previsao2)

# Verificar se o modelo de regressão está bom. Quanto mais próximo de 1 (100%), melhor.
score = regressor.score(X,y)
print(score)

# 7) Caso seja necessário, podemos também visualizar os valores residuais, ou seja, os pontos que estão distantes da reta vermelha traçada pelo gráfico de regressão

visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof()