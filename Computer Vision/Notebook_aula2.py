# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:13:10 2022

@author: elgom
"""
#Baixar e instalar o trainer: https://amin-ahmadi.com/cascade-trainer-gui/

#pip install icrawler

#Captura imagens positivas
from icrawler.builtin import BingImageCrawler


classes=['cats images'] 
total_max=100

for c in classes:
    bing_crawler=BingImageCrawler(storage={'root_dir':f'p'})
    bing_crawler.crawl(keyword=c,filters=None,max_num=total_max,offset=0,file_idx_offset='auto')
    
    
#Captura imagens negativas
from icrawler.builtin import BingImageCrawler         
classes=['trees','roads','Human faces']
total_max=50
for c in classes:
    bing_crawler=BingImageCrawler(storage={'root_dir':f'n'}) 
    bing_crawler.crawl(keyword=c,filters=None,max_num=total_max,offset=0,file_idx_offset='auto') 
    
    
#Depois de treinado o classificador

import cv2 # OpenCV

imagem = cv2.imread('cat.jpg')



def imshow(img,proporcao):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', int(img.shape[1]*proporcao), int(img.shape[0]*proporcao))
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imshow(imagem,0.5)




detector_face = cv2.CascadeClassifier('cascade.xml')

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
imshow(imagem_cinza,0.5)


deteccoes = detector_face.detectMultiScale(imagem_cinza)
deteccoes = detector_face.detectMultiScale(imagem_cinza, minSize=(110,110))

deteccoes

len(deteccoes)

for (x, y, l, a) in deteccoes:
  #print(x, y, l, a)
  cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2)
  
  
imshow(imagem,0.5)