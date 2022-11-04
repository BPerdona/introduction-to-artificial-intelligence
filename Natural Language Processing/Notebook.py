# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:18:58 2022

@author: elgomes
"""

# Processamento de linguagem natural com Python

## Importação das bibliotecas

#pip install spacy==2.3.8
#python -m spacy download pt

import spacy
spacy.__version__

import bs4 as bs
import urllib.request
import nltk



## Marcação POS
'''
- POS (part-of-speech) atribui para as palavras partes da fala, como substantivos, adjetivos, verbos
- Importante para a detecção de entidades no texto, pois primeiro é necessário saber o que o texto contém
- Documentação: https://v2.spacy.io/api/annotation#pos-tagging
- Português: https://www.sketchengine.eu/portuguese-freeling-part-of-speech-tagset/
'''

pln = spacy.load('pt_core_news_sm')


documento = pln('Estou aprendendo processamento de linguagem natural no IFSC de Canoinhas.')

type(documento)

for token in documento:
  print(token.text, token.pos_)
  
  

## Lematização

for token in documento:
  print(token.text, token.lemma_)

doc = pln('encontrei encontraram encontrarão encontrariam')
[token.lemma_ for token in doc]


##Stemização
nltk.download('rslp')

stemmer = nltk.stem.RSLPStemmer()
stemmer.stem('aprender')

for token in documento:
  print(token.text, token.lemma_, stemmer.stem(token.text))



## Carregamento dos textos

dados = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')


dados = dados.read()
dados

dados_html = bs.BeautifulSoup(dados, 'lxml')
dados_html

paragrafos = dados_html.find_all('p')

len(paragrafos)
paragrafos[0]

#Apaga os dois primeiros parágrafos
paragrafos.pop(0)
paragrafos.pop(0)

paragrafos[0].text

conteudo = ''
for p in paragrafos:
  conteudo += p.text

conteudo

conteudo = conteudo.lower()
conteudo

## Buscas em textos com spaCy


string = 'turing'
token_pesquisa = pln(string)


from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(pln.vocab)
matcher.add('SEARCH', None, token_pesquisa)

doc = pln(conteudo)
matches = matcher(doc)
matches

doc[3014:3015], doc[3014-5:3015+5]

doc[8944:8945], doc[8944-5:8945+5]




## Extração de entidades nomeadas
'''
- NER (Named-Entity Recognition)
- Encontrar e classificar entidades no texto, dependendo da base de dados que foi utilizada para o treinamento (pessoa, localização, empresa, numéricos)
- Siglas: https://v2.spacy.io/api/annotation#named-entities
'''


for entidade in doc.ents:
  print(entidade.text, entidade.label_)

from spacy import displacy
from pathlib import Path

html = displacy.render(doc, style="ent", page=True)
output_path = Path("NER.html")
output_path.open("w", encoding="utf-8").write(html)



## Nuvem de palavras e stop words
# pip install Pillow
# pip install wordcloud


from matplotlib.colors import ListedColormap

color_map = ListedColormap(['orange', 'green', 'red', 'magenta'])

from wordcloud import WordCloud
cloud = WordCloud(background_color = 'white', max_words = 100, colormap=color_map)

import matplotlib.pyplot as plt
cloud = cloud.generate(conteudo)
plt.figure(figsize=(15,15))
plt.imshow(cloud)
plt.axis('off')
plt.show()



from spacy.lang.pt.stop_words import STOP_WORDS
print(STOP_WORDS)

len(STOP_WORDS)

pln.vocab['caminhar'].is_stop

pln.vocab['usa'].is_stop

doc = pln(conteudo)
lista_token = []
for token in doc:
  lista_token.append(token.text)

print(lista_token)
len(lista_token)

sem_stop = []
for palavra in lista_token:
  if pln.vocab[palavra].is_stop == False:
    sem_stop.append(palavra)

print(sem_stop)
len(sem_stop)

cloud = cloud.generate(' '.join(sem_stop))
plt.figure(figsize=(15,15))
plt.imshow(cloud)
plt.axis('off')
plt.show()