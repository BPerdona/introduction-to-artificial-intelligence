# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:38:29 2022

@author: elgomes


# Classificação de textos com spaCy

# Etapa 1: Importação e instalação das bibliotecas
"""

#pip install spacy==2.3.8
#python -m spacy download pt

import spacy
spacy.__version__

import pandas as pd
import string
import random
import seaborn as sns
import numpy as np


"""# Etapa 2: Carregamento da base de dados"""

base_dados = pd.read_csv('base_treinamento.txt', encoding = 'utf-8')

base_dados.shape

base_dados.head()

base_dados.tail()

sns.countplot(base_dados['emocao'], label = 'Contagem');

"""# Etapa 3: Função para pré-processamento dos textos"""

pontuacoes = string.punctuation
pontuacoes

from spacy.lang.pt.stop_words import STOP_WORDS
stop_words = STOP_WORDS

print(stop_words)

len(stop_words)

pln = spacy.load('pt_core_news_sm')

pln

def preprocessamento(texto):
  texto = texto.lower()
  documento = pln(texto)
  
  lista = []
  for token in documento:
    #lista.append(token.text)
    lista.append(token.lemma_)

  lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
  lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

  return lista

teste = preprocessamento('Estou aPrendendo 1 10 23 processamento de linguagem natural no IFSC')
teste


"""# Etapa 4: Pré-processamento da base de dados

### Limpeza dos textos
"""

base_dados.head(10)

base_dados['texto'] = base_dados['texto'].apply(preprocessamento)

base_dados.head(10)



"""### Tratamento da classe"""

exemplo_base_dados = [["este trabalho é agradável", {"ALEGRIA": True, "MEDO": False}],
                      ["este lugar continua assustador", {"ALEGRIA": False, "MEDO": True}]]

type(exemplo_base_dados)

exemplo_base_dados[0]

exemplo_base_dados[0][0]

exemplo_base_dados[0][1]

type(exemplo_base_dados[0][1])

base_dados_final = []
for texto, emocao in zip(base_dados['texto'], base_dados['emocao']):
  #print(texto, emocao)
  if emocao == 'alegria':
    dic = ({'ALEGRIA': True, 'MEDO': False})
  elif emocao == 'medo':
    dic = ({'ALEGRIA': False, 'MEDO': True})

  base_dados_final.append([texto, dic.copy()])

len(base_dados_final)

base_dados_final[0]

base_dados_final[0][0]

base_dados_final[0][1]

type(base_dados_final[0][1])

base_dados_final


"""# Etapa 5: Criação do classificador"""

modelo = spacy.blank('pt')
categorias = modelo.create_pipe("textcat")
categorias.add_label("ALEGRIA")
categorias.add_label("MEDO")
modelo.add_pipe(categorias)
historico = []

modelo.begin_training()
for epoca in range(200):
  random.shuffle(base_dados_final)
  losses = {}
  for batch in spacy.util.minibatch(base_dados_final, 30):
    textos = [modelo(texto) for texto, entities in batch]
    annotations = [{'cats': entities} for texto, entities in batch]
    modelo.update(textos, annotations, losses=losses)
  if epoca % 100 == 0:
    print(losses)
    historico.append(losses)

historico_loss = [] #Para pegar apenas os valores
for i in historico:
  historico_loss.append(i.get('textcat'))

historico_loss = np.array(historico_loss)
historico_loss

import matplotlib.pyplot as plt
plt.plot(historico_loss)
plt.title('Progressão do erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')

#Salvar o modelo
modelo.to_disk("modelo")




"""# Etapa 6: Testes com uma frase"""

modelo_carregado = spacy.load("modelo")
modelo_carregado

texto_positivo = 'eu adoro cor dos seus olhos'

texto_positivo = preprocessamento(texto_positivo)
texto_positivo

previsao = modelo_carregado(texto_positivo)
previsao.cats

texto_negativo = 'estou com medo dele'
previsao = modelo_carregado(preprocessamento(texto_negativo))
previsao.cats

texto_alt = 'sai correndo por conta da chuva'
previsao = modelo_carregado(preprocessamento(texto_alt))
previsao.cats




"""# Etapa 7: Avaliação do modelo

## Avaliação na base de treinamento
"""

previsoes = []
for texto in base_dados['texto']:
  #print(texto)
  previsao = modelo_carregado(texto)
  previsoes.append(previsao.cats)

previsoes

previsoes_final = []
for previsao in previsoes:
  if previsao['ALEGRIA'] > previsao['MEDO']:
    previsoes_final.append('alegria')
  else:
    previsoes_final.append('medo')

previsoes_final = np.array(previsoes_final)

previsoes_final

respostas_reais = base_dados['emocao'].values
respostas_reais

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_score(respostas_reais, previsoes_final)

cm = confusion_matrix(respostas_reais, previsoes_final)
cm




"""## Avaliação na base de teste"""

base_dados_teste = pd.read_csv('base_teste.txt', encoding = 'utf-8')

base_dados_teste.head()

base_dados_teste['texto'] = base_dados_teste['texto'].apply(preprocessamento)

base_dados_teste.head()

previsoes = []
for texto in base_dados_teste['texto']:
  #print(texto)
  previsao = modelo_carregado(texto)
  previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
  if previsao['ALEGRIA'] > previsao['MEDO']:
    previsoes_final.append('alegria')
  else:
    previsoes_final.append('medo')

previsoes_final = np.array(previsoes_final)

respostas_reais = base_dados_teste['emocao'].values

accuracy_score(respostas_reais, previsoes_final)

cm = confusion_matrix(respostas_reais, previsoes_final)
cm
