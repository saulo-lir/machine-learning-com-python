import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

base = pd.read_csv('../data_bases/plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

# Regressão linear simples
regressor1 = LinearRegression()
regressor1.fit(X, y)
score1 = regressor1.score(X, y)
print(score1)

previsao1 = regressor1.predict(np.array(40).reshape(1, -1))
print(previsao1)

plt.scatter(X, y)
plt.plot(X, regressor1.predict(X), color = 'red')
plt.title('Regressão linear')
plt.xlabel('Idade')
plt.ylabel('Custo')
plt.show()

# Regressão linear polinomial
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly, y)
score2 = regressor2.score(X_poly, y)
print(score2)

previsao2 = regressor2.predict(poly.transform(np.array(40).reshape(1, -1)))
print(previsao2)

plt.scatter(X, y)
plt.plot(X, regressor2.predict(poly.fit_transform(X)), color = 'red')
plt.title('Regressão polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')
plt.show()