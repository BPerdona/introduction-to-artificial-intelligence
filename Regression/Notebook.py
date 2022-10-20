import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Importando e separando
base_plano_saude = pd.read_csv("plano_saude.csv")
X_plano_saude = base_plano_saude.iloc[:,0].values
y_plano_saude = base_plano_saude.iloc[:,1].values

# Coeficiente de relação
np.corrcoef(X_plano_saude, y_plano_saude)

# Colocar no formato de matriz
X_plano_saude = X_plano_saude.reshape(-1,1)

################### LINEAR REGRESSION ###############

# Utilizando regressão
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_plano_saude, y_plano_saude)

# Prevendo
previsoes = regressor.predict(X_plano_saude)

# Criando grafico
grafico = px.scatter(x=X_plano_saude.ravel(), y=y_plano_saude)
grafico.add_scatter(x=X_plano_saude.ravel(), y=previsoes, name='Regressão')
grafico.write_html("grafico_0.html")

# Prevendo pessoal
regressor.predict([[22]])


## Base de dados das casas (Utilizando uma coluna apenas)
base_casas = pd.read_csv('house_prices.csv')

# Criando heat Map
figura = plt.figure(figsize=(20,20))
sns.heatmap(base_casas.corr(), annot=True)

# Dividindo colunas
X_casas = base_casas.iloc[:,5:6].values
y_casas = base_casas.iloc[:,2].values

# Dividindo em teste
from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_casas_treinamento, y_casas_treinamento)

# Score
regressor.score(X_casas_treinamento, y_casas_treinamento)
regressor.score(X_casas_teste, y_casas_teste)

# Previsões
previsoes = regressor.predict(X_casas_treinamento)

# Graficos
grafico = px.scatter(x=X_casas_treinamento.ravel(), y=previsoes)
grafico.write_html("grafico_1.html")

# Graficos unidos
grafico1 = px.scatter(x=X_casas_treinamento.ravel(), y=y_casas_treinamento)
grafico2 = px.line(x=X_casas_treinamento.ravel(), y=previsoes)
grafico2.data[0].line.color = 'red'
grafico3 = go.Figure(data=grafico1.data+grafico2.data)
grafico3.write_html('grafico3.html')

previsoes_teste = regressor.predict(X_casas_teste)

# Gerando média de erro
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_casas_teste, previsoes_teste)
mean_squared_error(y_casas_teste, previsoes_teste)


# Base Casas com multiplas colunas
X_casas = base_casas.iloc[:,3:19].values
y_casas = base_casas.iloc[:,2].values

# Dividindo em teste
from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)

# Criando e alimento regressor
regressor = LinearRegression()
regressor.fit(X_casas_treinamento, y_casas_treinamento)

# Prevendo
previsoes_teste = regressor.predict(X_casas_teste)

# Gerando média de erro
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_casas_teste, previsoes_teste)
mean_squared_error(y_casas_teste, previsoes_teste)

################### POLYNOMIAL ###################

# Base Casas com multiplas colunas Polynomial
from sklearn.preprocessing import PolynomialFeatures

## Base de dados das casas (Utilizando uma coluna apenas)
base_casas = pd.read_csv('house_prices.csv')

X_casas = base_casas.iloc[:,3:19].values
y_casas = base_casas.iloc[:,2].values

# Dividindo em teste
from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)

# Criando polinômio
poly = PolynomialFeatures(degree=2)

# Criando valores com polinomios (2º cria o quadrado e o cubo de cada valor)
X_casas_treinamento_poly = poly.fit_transform(X_casas_treinamento)
X_casas_teste_poly = poly.fit_transform(X_casas_teste)

# Criando e alimento regressor
regressor = LinearRegression()
regressor.fit(X_casas_treinamento_poly, y_casas_treinamento)

# Prevendo
previsoes_teste = regressor.predict(X_casas_teste_poly)

# Gerando média de erro
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_casas_teste, previsoes_teste)
mean_squared_error(y_casas_teste, previsoes_teste)


####### Árvore de decisão ########

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

regressor.fit(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_treinamento,y_casas_treinamento)

regressor.score(X_casas_teste,y_casas_teste)

previsoes_teste = regressor.predict(X_casas_teste)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(y_casas_teste, previsoes_teste)


######## Random ForestRegressor ##########

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_treinamento,y_casas_treinamento)

regressor.score(X_casas_teste,y_casas_teste)

previsoes_teste = regressor.predict(X_casas_teste)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(y_casas_teste, previsoes_teste)


######## SVRRegressor #########

from sklearn.svm import SVR

regressor = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)

regressor.fit(X_casas_treinamento, y_casas_treinamento)

regressor.score(X_casas_treinamento,y_casas_treinamento)

regressor.score(X_casas_teste,y_casas_teste)

previsoes_teste = regressor.predict(X_casas_teste)

from sklearn.metrics import mean_absolute_error,mean_squared_error

######## ForestRegressor

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(X_casas_treinamento, y_casas_treinamento)


regressor.score(X_casas_treinamento,y_casas_treinamento)

regressor.score(X_casas_teste,y_casas_teste)

previsoes_teste = regressor.predict(X_casas_teste)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(y_casas_teste, previsoes_teste)


### SVRRegressor

from sklearn.svm import SVR

regressor = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)

regressor.fit(X_casas_treinamento, y_casas_treinamento)


regressor.score(X_casas_treinamento,y_casas_treinamento)

regressor.score(X_casas_teste,y_casas_teste)

previsoes_teste = regressor.predict(X_casas_teste)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(y_casas_teste, previsoes_teste)