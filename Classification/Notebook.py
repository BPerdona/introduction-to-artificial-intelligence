import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### Exploração dos dados
base_credit = pd.read_csv('credit_data.csv')

base_credit.head(10)
base_credit.tail(10)
base_credit.describe() #Informações uteis

base_credit[base_credit['income']>=69995]
base_credit['income']>=69995

### Visualização de dados
np.unique(base_credit['default'], return_counts=True)

sns.countplot(x=base_credit['default'])

plt.hist(x=base_credit['age'])

plt.hist(x=base_credit['income'])

plt.hist(x=base_credit['loan'])

sns.pairplot(base_credit, vars=['age','income'], hue='default')

sns.pairplot(base_credit, vars=['age','income', 'loan'], hue='default')

### Tratamento de valores inconsistentes
base_credit.loc[base_credit['age'] < 0]

### Apagando coluna inteira, vale a pena?
base_credit2 = base_credit.drop('age', axis=1)

### Apagar somente os valores inconsistentes?
base_credit3 = base_credit.drop(base_credit[base_credit['age']<0].index)

### Preencher manualmente? Prencher com a média?
base_credit.mean()

base_credit['age'][base_credit['age'] > 0].mean()

base_credit.loc[base_credit['age']<0, 'age'] = 40.927

### Tratamento de valores faltantes
base_credit.isnull().sum()

base_credit.loc[pd.isnull(base_credit['age'])]

base_credit.loc[pd.isnull(base_credit['age']), 'age'] = 40.937

### Divisão entre previsores e classe - Pode-se colocar as colunas em [] ao inves de 1:4

# Previsores
X_credit = base_credit.iloc[:,1:4].values

# Classes
Y_credit = base_credit.iloc[:, 4].values


### Escalonamento dos valores

# Min
X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min()

# Max
X_credit[:,0].max(), X_credit[:,1].max(), X_credit[:,2].max()

# Normalização
from sklearn.preprocessing import MinMaxScaler
scaler_credit = MinMaxScaler()

X_credit = scaler_credit.fit_transform(X_credit)

### Aula 02 - Senso

# Exploração dos dados csv
base_census = pd.read_csv('census.csv')

#Descrevendo
base_census.describe()

#Verificando se tem nulls
base_census.isnull().sum()

#Visualização dos dados (Graficos)
np.unique(base_census['income'], return_counts=True)
sns.countplot(x=base_census['income'])
plt.hist(x=base_census['age'])
plt.hist(x=base_census['education-num'])
plt.hist(x=base_census['hour-per-week'])

#Trimap
import plotly.express as px

grafico = px.treemap(base_census, path=["workclass"])
grafico.write_html('grafico0.html')

grafico = px.treemap(base_census, path=["workclass", 'age'])
grafico.write_html('grafico1.html')

grafico = px.treemap(base_census, path=["occupation", 'relationship', 'age'])
grafico.write_html('grafico2.html')

##Categorias paralelas
grafico = px.parallel_categories(base_census, dimensions=["occupation", 'relationship'])
grafico.write_html("grafico3.html")

grafico = px.parallel_categories(base_census, dimensions=["native-country", 'income'])
grafico.write_html("grafico4.html")

grafico = px.parallel_categories(base_census, dimensions=["workclass", 'occupation', 'income'])
grafico.write_html("grafico5.html")

grafico = px.parallel_categories(base_census, dimensions=["education", 'income'])
grafico.write_html("grafico6.html")

###Divisão entre previsores e classe
X_census = base_census.iloc[:,0:14].values
Y_census = base_census.iloc[:,14].values

##Tratamento de atributos categóricos
#LabelEncoder ->

#P M G GG XG
#0 1 2  3  4
                                
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

indices = [1,3,5,6,7,8,9,13]

for i in indices:    
    X_census[:,i] = label_encoder.fit_transform(X_census[:,i])
    
    
### OneHotEncoder
#Carro

#Gol Pálio Uno
# 1    2    3

#Gol    1   0   0
#Pálio  0   1   0
#Uno    0   0   1

len(np.unique(base_census['workclass'])) # -> 9
# 1 0 0 0 0 0 0 0 0
# 0 1 0 0 0 0 0 0 0
# 0 0 1 0 0 0 0 0 0
# ...

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(),indices)], remainder='passthrough')

X_census = onehotencoder_census.fit_transform(X_census).toarray()

X_census.shape


#Escalonamento de valores
from sklearn.preprocessing import StandardScaler
scaler_sensus = StandardScaler()
X_census = scaler_sensus.fit_transform(X_census)


#Divisão das bases em treinamento e teste
from sklearn.model_selection import train_test_split

X_census_treinamento, X_census_teste, Y_census_treinamento, Y_census_teste = train_test_split(X_census, Y_census, test_size=0.25, random_state=0)
X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste = train_test_split(X_credit, Y_credit, test_size=0.25, random_state=0)

#Salvar variáveis
import pickle

with open("census.pkl", mode="wb") as f:
    pickle.dump([X_census_treinamento, X_census_teste, Y_census_treinamento, Y_census_teste], f)

with open("credit.pkl", mode="wb") as f:
    pickle.dump([X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste], f)

#### Aula 3

# Abrindo variaveis
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f) 

#Verificando
X_credit_treinamento.shape, y_credit_treinamento.shape
X_credit_teste.shape, y_credit_teste.shape


#Aprendendo
from sklearn.naive_bayes import GaussianNB
naive_credit = GaussianNB()
naive_credit.fit(X_credit_treinamento, y_credit_treinamento)

#Fazendo os teste
previsoes_credit = naive_credit.predict(X_credit_teste)


#Importando metricas para ver a importancia
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Comparando a taxa de acerto
accuracy_score(y_credit_teste, previsoes_credit)

#Matriz de confusao
confusion_matrix(y_credit_teste, previsoes_credit)

#Matriz confusa com tijolo amarelo
from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(naive_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)

#Relatório geral
print(classification_report(y_credit_teste, previsoes_credit))


## Base census

#Abrindo arquivo
with open("census.pkl", 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste=pickle.load(f)
    
# Verificando integridade
X_census_treinamento.shape, y_census_treinamento.shape
X_census_teste.shape, y_census_teste.shape
    
naive_census = GaussianNB()
naive_census.fit(X=X_census_treinamento, y=y_census_treinamento)   
    
previsoes_census = naive_census.predict(X_census_teste)
    
accuracy_score(y_census_teste, previsoes_census)


cm = ConfusionMatrix(naive_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)


print(classification_report(y_census_teste, previsoes_census))

########## Aula - 5 ##############

# Arvore de decisão
from sklearn.tree import DecisionTreeClassifier   

##Base de dados - Credito
import pickle    

with open('risco_credito.pkl', 'rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)
    
#Criando instancia
arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')

#Criando arvore
arvore_risco_credito.fit(X_risco_credito, y_risco_credito)

#Imprimindo arvore de risco
arvore_risco_credito.feature_importances_

#Mostrando as classes
arvore_risco_credito.classes_

#Criando novas previsões
previsao = arvore_risco_credito.predict([[0,0,1,2],[2,0,0,0]])


from sklearn import tree
import matplotlib.pyplot as plt

previsores = ['historia', 'divida', 'garantias', 'renda']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(arvore_risco_credito, feature_names=previsores, class_names=arvore_risco_credito.classes_, filled=True)

figura.savefig('risco_credito_tree.pdf')

#Base de dados credit_data

with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f) 


#Criando a arvore de decisao
arvore_credit = DecisionTreeClassifier(criterion="entropy")
arvore_credit.fit(X_credit_treinamento,y_credit_treinamento)
previsoes = arvore_credit.predict(X_credit_teste)

#Criando relatorios
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_credit_teste, previsoes)


previsores = ['income', 'age', 'loan']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
tree.plot_tree(arvore_credit, feature_names=previsores, class_names=['0','1'], filled=True)

figura.savefig('risco_credit_tree.pdf')


##Base census
with open("census.pkl", 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste=pickle.load(f)

arvore_census = DecisionTreeClassifier(criterion="entropy")
arvore_census.fit(X_census_treinamento,y_census_treinamento)
previsoes = arvore_census.predict(X_census_teste)

accuracy_score(y_census_teste, previsoes)
#0.8155

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(arvore_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

print(classification_report(y_census_teste, previsoes))

##### Random Forest ######
from sklearn.ensemble import RandomForestClassifier

#Base credit data
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f) 

#Criando random forest
random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
random_forest_credit.fit(X_credit_treinamento, y_credit_treinamento)
previsoes = random_forest_credit.predict(X_credit_teste)
accuracy_score(y_credit_teste, previsoes)
print(classification_report(y_credit_teste, previsoes))
#98.4

##Base census
with open("census.pkl", 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste=pickle.load(f)

#Criando random forest census
random_forest_credit = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
random_forest_credit.fit(X_census_treinamento, y_census_treinamento)
previsoes = random_forest_credit.predict(X_census_teste)
accuracy_score(y_census_teste, previsoes)
print(classification_report(y_census_teste, previsoes))
#84.98


###### AULA 6 ######
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f) 

#Criando KNN
knn_credit = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn_credit.fit(X_credit_treinamento, y_credit_treinamento)

#Previsoes
previsoes = knn_credit.predict(X_credit_teste)

accuracy_score(y_credit_teste, previsoes)
print(classification_report(y_credit_teste,previsoes))
#98.4

with open("census.pkl", 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste=pickle.load(f)

knn_census = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = knn_census.predict(X_census_teste)

accuracy_score(y_census_teste, previsoes)
print(classification_report(y_census_teste,previsoes))
#82.0

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(knn_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

####### SVM #########
from sklearn.svm import SVC

with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f) 

#Criando SVM
svm_credit = SVC(C=5, kernel='rbf')
svm_credit.fit(X_credit_treinamento, y_credit_treinamento)

#Previsoes
previsoes = svm_credit.predict(X_credit_teste)

accuracy_score(y_credit_teste, previsoes)
print(classification_report(y_credit_teste,previsoes))
#98.6

with open("census.pkl", 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste=pickle.load(f)

svm_census = SVC(C=5, kernel='rbf')
svm_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = svm_census.predict(X_census_teste)

accuracy_score(y_census_teste, previsoes)
print(classification_report(y_census_teste,previsoes))
#85.06

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(svm_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)


#### REDE NEURAL ####
from sklearn.neural_network import MLPClassifier


with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f) 

#Criando rede neural
rede_neural_credit = MLPClassifier(max_iter=1000, verbose=True, random_state=0, n_iter_no_change=10)
rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)

#Previsoes
previsoes = rede_neural_credit.predict(X_credit_teste)

accuracy_score(y_credit_teste, previsoes)
print(classification_report(y_credit_teste,previsoes))
#99.6

with open("census.pkl", 'rb') as f:
    X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste=pickle.load(f)

rede_neural_census =  MLPClassifier(max_iter=100, verbose=True, random_state=0, n_iter_no_change=10)
rede_neural_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = rede_neural_census.predict(X_census_teste)

accuracy_score(y_census_teste, previsoes)
print(classification_report(y_census_teste,previsoes))
#84.09

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(rede_neural_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)






