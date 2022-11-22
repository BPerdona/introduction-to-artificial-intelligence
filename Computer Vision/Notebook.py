# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:49:10 2022

@author: elgom
"""

# Visão computacional

## Detecção de faces

#pip install opencv-contrib-python 

import cv2 # OpenCV

imagem = cv2.imread('workplace-1245776_1920.jpg')



def imshow(img,proporcao):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', int(img.shape[1]*proporcao), int(img.shape[0]*proporcao))
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imshow(imagem,0.5)


#Para criar um classificador deve ser usado um software específico
#https://amin-ahmadi.com/cascade-trainer-gui/


detector_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
imshow(imagem_cinza,0.5)

# Detecção
deteccoes = detector_face.detectMultiScale(imagem_cinza)

# Sem fantasmas
# deteccoes = detector_face.detectMultiScale(imagem_cinza, scaleFactor=1.3, minSize=(30,30))

deteccoes

len(deteccoes)

for (x, y, l, a) in deteccoes:
  #print(x, y, l, a)
  cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2)
  
  
imshow(imagem,0.5)


## Detecção do corpo

image = cv2.imread('pessoas.jpg')
imshow(image,1)

detector_corpo = cv2.CascadeClassifier('fullbody.xml')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = detector_corpo.detectMultiScale(image_gray)
#detections = detector_corpo.detectMultiScale(image_gray, scaleFactor=1.1, minSize=(50,50))
print(len(detections))
print(detections)
for (x, y, l, a) in detections:
  cv2.rectangle(image, (x, y), (x + l, y + a), (0,255,0), 2)
imshow(image,1)



## Reconhecimento facial

### Treinamento
#pip install --upgrade Pillow
from PIL import Image
import numpy as np


import os
os.listdir('yalefaces/train')

def dados_imagem():
  caminhos = [os.path.join('yalefaces/train', f) for f in os.listdir('yalefaces/train')]
  faces = []
  ids = []
  for caminho in caminhos:
    imagem = Image.open(caminho).convert('L')
    imagem_np = np.array(imagem, 'uint8')
    id = int(os.path.split(caminho)[1].split('.')[0].replace('subject', ''))
    ids.append(id)
    faces.append(imagem_np)
  return np.array(ids), faces

ids, faces = dados_imagem()

print(ids)

print(faces[0])

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

### Classificação

reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read('classificadorLBPH.yml')

imagem_teste = 'yalefaces/test/subject10.sad.gif'

imagem = Image.open(imagem_teste).convert('L')
imagem_np = np.array(imagem, 'uint8')
print(imagem_np)

idprevisto, confianca = reconhecedor.predict(imagem_np)
idprevisto
print(confianca)

idcorreto = int(os.path.split(imagem_teste)[1].split('.')[0].replace('subject', ''))
idcorreto

cv2.putText(imagem_np, 'P: ' + str(idprevisto), (12,96), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.putText(imagem_np, 'C: ' + str(idcorreto), (12,116), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
imshow(imagem_np,1)


#Rastreamento de objetos
import cv2 # OpenCV

rastreador = cv2.TrackerCSRT_create()
video = cv2.VideoCapture('rua.mp4')
ok, frame = video.read()

bbox = cv2.selectROI(frame) #inicializa o boundBox
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(bbox)

ok = rastreador.init(frame,bbox)

while True:
    ok, frame = video.read()
    if not ok:
        break
    
    ok,bbox = rastreador.update(frame)
    #print(bbox)
    
    if ok:
        (x,y,w,h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 2, 1)
    else:
        cv2.putText(frame,'Falha no rastreamento', (100,800), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)
        
    cv2.imshow('Rastreando', frame)
    if cv2.waitKey(1) & 0XFF == 27:
        cv2.destroyAllWindows()
        break