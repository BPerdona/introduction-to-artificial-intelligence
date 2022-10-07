## Naive_bayes
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Base de riscos
base_risco = pd.read_csv("risco_credito.csv")

# Divisão
X_risco_credito = base_risco.iloc[:,0:4].values
y_risco_credito = base_risco.iloc[:,4].values

# Encoder
label_encoder_risco = LabelEncoder()
indices_risco = [0,1,2,3]
for i in indices_risco:
    X_risco_credito[:,i] = label_encoder_risco.fit_transform(X_risco_credito[:,i])


# Salvando com Pickle
with open("risco_credito.pkl", "wb") as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)


#Treinando
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, y_risco_credito)


#História 1: Boa(0), Divida: alta(0), garantias: nenhuma(1), renda: > 35 (2)
#História 2: Ruim(2), Divida: alta(0), garantias: adequada(0), renda: < 15 (0)

#Resultado
previsao = naive_risco_credito.predict([[0,0,1,2],[2,0,0,0]])


















