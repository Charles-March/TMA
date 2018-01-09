# -*- coding: utf-8 -*-
"""
Created on Tue Oct 06 18:37:39 2015

@author: Patricia
"""
#######To clear the working memory###########
def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]
#############################################
#clearall()    
    
import os        
import numpy as np
from skimage.color import rgb2gray
from skimage import data, measure,io
import skimage as sk
import matplotlib.pyplot as plt
from dct2 import dct2, idct2
import PIL as pil #pour utiliser la librairie d'écriture de fichier jpeg


plt.close('all')

   
#img = data.astronaut()

img=io.imread('Im3Comp100.jpg')
img_gray = rgb2gray(img)
img_gray=sk.img_as_float(img_gray)
plt.figure(1)
plt.imshow(img_gray,cmap='gray')
N=8
M=8
window_size = (N,M)
img_size=img_gray.shape

dctblock=np.zeros(img_size)

dctblock=dct2(img_gray)
       

plt.figure(3)
plt.imshow(np.log(1.0+dctblock),cmap='gray')

newim=np.zeros(img_size)
newim=idct2(dctblock)

psnr=measure.compare_psnr(img_gray,newim,1.0)
ssim=measure.compare_ssim(img_gray,newim)

plt.figure(4)
newim=np.ubyte(np.round(255.0*newim,0))
#remettre le type des données ici entre 0 et 255 donc uint8
plt.imshow(newim,cmap='gray')

fich=open('madct.dat','wb')
fich.write(np.reshape(newim,-1)) 
#on étend le tableau en 1D pour pouvoir enregistrer chaque octet
fich.close()



#pour sauver l'image en format jpeg pour une qualité voulue
monIm=pil.Image.fromarray(np.ubyte(np.round(255.0*img_gray,0)))
monIm.save('essai.jpeg',quality=20)
monImlu=pil.Image.open("essai.jpeg")
print( "taille= ",os.path.getsize("essai.jpeg"), "en octet")
print("compression =", 1.0*img_size[0]*img_size[1]/os.path.getsize("essai.jpeg"))

