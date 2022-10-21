import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# Formando 3 array (Estilo cubo) | Recomendado utilizar jpg ou bmp
img_arr = img.imread('paisagem.bmp')

plt.imshow(img_arr)

# Retirando as cores
(h,w,c) = img_arr.shape

# Criando um unico array contendo todos os pixeis e seu RGB
img2D = img_arr.reshape(h*w,c)

from sklearn.cluster import KMeans

# Instanciando o KMeans | Para aumentar a qualidade da imagem basta aumentar os clusters
kmeans_model = KMeans(n_clusters=6, init="random")

# Processando
clusters_labels = kmeans_model.fit_predict(img2D)

# Listando as instancias escolhidas como centroids
centroids = kmeans_model.cluster_centers_

# Arredondando os centroids
rgb_colors = centroids.round(0).astype(int)

# Pegando os resultados dos pixeis
labels = (kmeans_model.labels_)

# Criando o array tipo cubo com os clusters
img_quant = np.reshape(rgb_colors[clusters_labels],(h,w,c))

# Criando imagem
plt.imshow(img_quant)

# Pegando as labels como listas
labels=list(kmeans_model.labels_)

# Separando as cores
percent=[]
for i in range(len(centroids)):
  j=labels.count(i)
  j=j/(len(labels))
  percent.append(j)
print(percent)

# Criando figura
fig, ax = plt.subplots(1,3, figsize=(20,12))
ax[0].imshow(img_arr)
ax[0].set_title('Imagem Original')
ax[1].imshow(img_quant)
ax[1].set_title('Imagem Quantizada')
ax[2].pie(percent,colors=np.array(centroids/255),labels=np.arange(len(centroids)))
ax[2].set_title('Paleta de cores')


##### MESMO PROCESSAMENTO #####
 

# Utilizando o mesmo processamento mas otimizado com k-means++
kmeans_model = KMeans(n_clusters=6, init="k-means++")
clusters_labels = kmeans_model.fit_predict(img2D)
centroids = kmeans_model.cluster_centers_
rgb_colors = centroids.round(0).astype(int)
labels = (kmeans_model.labels_)
img_quant = np.reshape(rgb_colors[clusters_labels],(h,w,c))
plt.imshow(img_quant)
labels=list(kmeans_model.labels_)

percent=[]
for i in range(len(centroids)):
  j=labels.count(i)
  j=j/(len(labels))
  percent.append(j)
print(percent)

fig, ax = plt.subplots(1,3, figsize=(20,12))
ax[0].imshow(img_arr)
ax[0].set_title('Imagem Original')
ax[1].imshow(img_quant)
ax[1].set_title('Imagem Quantizada')
ax[2].pie(percent,colors=np.array(centroids/255),labels=np.arange(len(centroids)))
ax[2].set_title('Paleta de cores')

 
### Kmeans com inicialização de centroides


# Criando array contendo
centroides = []
centroides.append([34,23,6]) # Montanha
centroides.append([230,253,253]) # Ceus
centroides.append([69,169,184]) # Mar
centroides.append([252,228,202]) # Areia

centroides = np.array(centroides)

kmeans_model = KMeans(n_clusters=4, init=centroides)
clusters_labels = kmeans_model.fit_predict(img2D)
centroids = kmeans_model.cluster_centers_
rgb_colors = centroids.round(0).astype(int)
labels = (kmeans_model.labels_)
img_quant = np.reshape(rgb_colors[clusters_labels],(h,w,c))
plt.imshow(img_quant)
labels=list(kmeans_model.labels_)
percent=[]
for i in range(len(centroids)):
  j=labels.count(i)
  j=j/(len(labels))
  percent.append(j)
print(percent)

fig, ax = plt.subplots(1,3, figsize=(20,12))
ax[0].imshow(img_arr)
ax[0].set_title('Imagem Original')
ax[1].imshow(img_quant)
ax[1].set_title('Imagem Quantizada')
ax[2].pie(percent,colors=np.array(centroids/255),labels=np.arange(len(centroids)))
ax[2].set_title('Paleta de cores')


### Agrupando valores de compra para conseguir regras
## Atenção!!! Instalar esse pacote -> apyori
# pip install apyori


import pandas as pd
from apyori import apriori

## Base de dados mercado 1
base_mercado1 = pd.read_csv('mercado.csv', header = None)

# Criando um lista com uma lista dentro
transacoes = []
for i in range(len(base_mercado1)):
  transacoes.append([str(base_mercado1.values[i, j]) for j in range(base_mercado1.shape[1])])

# Gerando as regras com o apriori
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2)
resultados = list(regras)

resultados

len(resultados)

resultados[2]


A = []
B = []
suporte = []
confianca = []
lift = []

# Criando dataframe a partir do resultado do processamento
for resultado in resultados:
  s = resultado[1]
  result_rules = resultado[2]
  for result_rule in result_rules:
    a = list(result_rule[0])
    b = list(result_rule[1])
    c = result_rule[2]
    l = result_rule[3]
    A.append(a)
    B.append(b)
    suporte.append(s)
    confianca.append(c)
    lift.append(l)

# Alternando visualização ordenação dos dados
rules_df = pd.DataFrame({'A': A, 'B': B, 'suporte': suporte, 'confianca': confianca, 'lift': lift})
rules_df = rules_df.sort_values(by = 'lift', ascending = False)






