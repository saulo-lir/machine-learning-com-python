import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

base = pd.read_csv('../data_bases/house_prices.csv')

X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 0)

poly = PolynomialFeatures(degree = 2)
X_treinamento_poly = poly.fit_transform(X_treinamento)
X_teste_poly = poly.transform(X_teste)

regressor = LinearRegression()
regressor.fit(X_treinamento_poly, y_treinamento)
score = regressor.score(X_treinamento_poly, y_treinamento)
print(score)

previsoes = regressor.predict(X_teste_poly)
print(previsoes)

mae = mean_absolute_error(y_teste, previsoes)
print(mae)