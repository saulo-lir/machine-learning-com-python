import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

base = pd.read_csv('../data_bases/house_prices.csv')

# 1) Separar os atributos independentes dos dependentes
X = base.iloc[:, 3:19].values # Ao passar diversos atributos previsores, a biblioteca já saberá que se trata de regressão linear múltipla
y = base.iloc[:, 2].values

# 2) Separar os atributos de treinamento dos de teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)

# 3) Treinar o modelo de regressão linear
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)

# 4) Verificar se o modelo de regressão está bom. Quanto mais próximo de 1 (100%), melhor.
score = regressor.score(X_treinamento, y_treinamento)
print(score)

# TO DO: Verificar como plotar (e se é possível) um gráfico de regressão linear múltipla

# 5) Prever o preço das casas com os dados de teste
previsoes = regressor.predict(X_teste)

# 6) Verificar a diferença média de erro entre a previsão e os dados de teste com métodos da biblioteca sklearn.metrics
mae = mean_absolute_error(y_teste, previsoes)

score_2 = regressor.score(X_teste, y_teste)
print(score_2)

# 7) Visualizar os parâmetros da fórmula matemática de regressão linear
b0 = regressor.intercept_
b1 = regressor.coef_