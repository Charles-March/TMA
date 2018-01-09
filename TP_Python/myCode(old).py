"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

"""
- Réalisez un programme qui calcule la DCT d'une image. Vous pouvez dans un premier
temps prendre toute l'image dans un seul bloc. Visualisez l'image DCT, quelle conclusion
pouvez-vous faire sur la localisation et la répartition des données? Où se trouve l'essentiel de
l'information.
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
import PIL as pil #pour utiliser la librairie d'écriture de fichier jpeg


# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:08:35 2016

@author: ladretp
"""
#Definition of local functions

from scipy.fftpack import dct,idct
#___________________________________________________________
def dct2(x):
    return dct(dct(x,norm='ortho').T,norm='ortho').T
#_____________________________________________________________  

def idct2(x):
    return idct(idct(x,norm='ortho').T,norm='ortho').T


plt.close('all')

   
#img = data.astronaut()

img=io.imread('Im3Comp100.jpg')
img_gray = rgb2gray(img)
img_gray=sk.img_as_float(img_gray)
plt.figure(1)
plt.imshow(img_gray,cmap='gray')
plt.show()
N=8
M=8


window_size = (N,M)
img_size=img_gray.shape





range_x= int(img_size[0]/N)
range_y= int(img_size[1]/M)
blocksSize = (range_x,range_y,N,M)
dctblocks = np.zeros(blocksSize)
dctArray = np.zeros(blocksSize)
for i in range(range_x):
    for j in range(range_y):
        for k in range(N):
            for l in range(M):
                dctblocks[i][j][k][l]= img_gray[i*N + k][j*M + l]
        dctArray[i][j]=dct2(dctblocks[i][j])
        
ImgNew = np.zeros(img_size)        

for i in range(range_x):
    for j in range(range_y):
        for k in range(N):
            for l in range(M):
                ImgNew[i*N + k][j*M + l]=dctArray[i][j][k][l]

"""
dctblock=np.zeros(img_size)
dctblock=dct2(img_gray)
       
"""

blocseul = dctblocks[0][0]
blocDctSeul = dctArray[0][0]

plt.figure(3)
plt.imshow(np.log(1.0+ImgNew),cmap='gray')
plt.show()

newim=np.zeros(img_size)


finalImg = np.zeros(img_size)
for i in range(range_x):
    for j in range(range_y):
        tempimg = idct2(dctblocks[i][j])
        psnr=measure.compare_psnr(dctblocks[i][j],tempimg,1.0)
        ssim=measure.compare_ssim(dctblocks[i][j],tempimg)
        #tempimg=np.ubyte(np.round(255.0*tempimg,0))
        for k in range(N):
            for l in range(M):
                finalImg[i*N + k,j*M +l]=(tempimg[k][l])
                
        
        
#newim=idct2(dctblocks)
"""
psnr=measure.compare_psnr(img_gray,newim,1.0)
ssim=measure.compare_ssim(img_gray,newim)
newim=np.ubyte(np.round(255.0*newim,0))
"""
plt.figure(4)

#remettre le type des données ici entre 0 et 255 donc uint8
plt.imshow(finalImg,cmap='gray')
plt.show()

fich=open('madct.dat','wb')
fich.write(np.reshape(finalImg,-1)) 
#on étend le tableau en 1D pour pouvoir enregistrer chaque octet
fich.close()



#pour sauver l'image en format jpeg pour une qualité voulue
monIm=pil.Image.fromarray(np.ubyte(np.round(255.0*img_gray,0)))
monIm.save('essai.jpeg',quality=20)
monImlu=pil.Image.open("essai.jpeg")
print( "taille= ",os.path.getsize("essai.jpeg"), "en octet")
print("compression =", 1.0*img_size[0]*img_size[1]/os.path.getsize("essai.jpeg"))

