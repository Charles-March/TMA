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


def Q(i,j,compression):
    return (1+((1+i+j)*compression))

def Fq(f,u,v,compression):
    return (np.round(f(u,v)/Q(u,v,compression)))

"""
@param 
    N*M : size of the blocks
    imgPath : local path to the img
@return
    dctArray : a N*M*(img_size[0]/N)*(img_size[1]/M) array corresponding to the dct of each N*M blocks of the picture.
"""
def dctBlocks(N,M,imgPath):
    img=io.imread(imgPath)
    img_gray = rgb2gray(img)
    img_gray=sk.img_as_float(img_gray)
    plt.figure(1)
    plt.imshow(img_gray,cmap='gray')
    plt.show()
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
            
    return (dctArray,img_size)
"""
@param
    dctArray : a N*M*(img_size[0]/N)*(img_size[1]/M) array corresponding to the dct of each N*M blocks of the picture.
    N*M : size of each block
    img_size : starting size of the img
@return
    finalImg : the img uncompressed
"""
def recompression(dctArray,N,M,img_size):
    finalImg = np.zeros(img_size)
    range_x= int(img_size[0]/N)
    range_y= int(img_size[1]/M)
    for i in range(range_x):
        #   temping = np.zeros((N,M))
        for j in range(range_y):
            tempimg = idct2(dctArray[i][j])
            #psnr=measure.compare_psnr(dctblocks[i][j],tempimg,1.0)
            #ssim=measure.compare_ssim(dctblocks[i][j],tempimg)
            #tempimg=np.ubyte(np.round(255.0*tempimg,0))
            for k in range(N):
                for l in range(M):
                    finalImg[i*N + k][j*M +l]=(tempimg[k][l]) 
    return finalImg

"""
@param
    dctArray : a N*M*(img_size[0]/N)*(img_size[1]/M) array corresponding to the dct of each N*M blocks of the picture.
    N*M : size of each block
    img_size : starting size of the img
@return
    dctImg : the img which represent all the dctblocks in one img
"""
def concatDct(dctArray,N,M,img_size):
    dctImg = np.zeros(img_size)        
    range_x= int(img_size[0]/N)
    range_y= int(img_size[1]/M)
    for i in range(range_x):
        for j in range(range_y):
            for k in range(N):
                for l in range(M):
                    dctImg[i*N + k][j*M + l]=dctArray[i][j][k][l]
    return dctImg

"""
create a new file called file_name and write inside the content of file
@param
    file : the content to write
    file_name : the name of the file that will be created / overwritted
"""
def exportfile(file,file_name):
    fich=open(file_name,'wb')
    fich.write(file) 
    fich.close()
    
def exportInBinary(file,file_name):
    fich=open(file_name)
    fich.write(np.reshape(file,-1)) 
    fich.close()
    
"""
print the file defined by imgPath
cut the picture defined by imgPath into multiples N*M blocks
calculate the dct of each block
print the dctblocks
recombine the dctblocks into the base picture and print the new picture
store the dctblocks in a file called 'dct4.dat' for a 4*4 cut
store the new picture in a file called 'essai4.jpeg' for a 4*4 cut

@param
    N*M : size of each block
    imgPath : the local path to the img
"""

def compression(N,M,imgPath):
    (dctArray,img_size)=dctBlocks(N,M,imgPath)
    ImgNew = concatDct(dctArray,N,M,img_size)
    exportfile(ImgNew,"dct"+str(N)+".dat")
    plt.figure(3)
    print("taille : "+str(N) +" "+str(M))
    plt.imshow(np.log(1.0+ImgNew),cmap='gray')
    plt.show()
    finalImg =recompression(dctArray,N,M,img_size)
    plt.figure(4)
    
 
    #remettre le type des données ici entre 0 et 255 donc uint8
    plt.imshow(finalImg,cmap='gray')
    plt.show()
    
    fich=open('madct'+str(N)+'.dat','wb')
    fich.write(np.reshape(finalImg,-1)) 
    #on étend le tableau en 1D pour pouvoir enregistrer chaque octet
    fich.close()
    
    #pour sauver l'image en format jpeg pour une qualité voulue
    finalImg=pil.Image.fromarray(np.ubyte(np.round(255.0*finalImg,0)))
    finalImg.save('essai'+str(N)+'.jpeg')
    monImlu=pil.Image.open("essai"+str(N)+'.jpeg')
    print( "taille= ",os.path.getsize("essai"+str(N)+'.jpeg'), "en octet")
    print("compression =", 1.0*img_size[0]*img_size[1]/os.path.getsize("essai"+str(N)+'.jpeg'))

compression(4,4,"horse.bmp")
compression(8,8,"horse.bmp")
compression(16,16,"horse.bmp")
compression(64,64,"horse.bmp")

