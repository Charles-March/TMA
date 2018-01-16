"""
editeur de Spyder

Ceci est un script temporaire.
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

"""
@return
    value : (1+((1+i+j)*compression))
"""

def Q(i,j,compression):
    return (1+((1+i+j))*compression)

"""
@param
    f : a function that will be applicated with f(block,u,v)
    u,v : position of a point in block
    compression : compression rate
    block : where the value should be stored (block[u][v])
@return
    value : return np.round((f(block,u,v)/Q(u,v,compression)))
"""
def Fq(f,u,v,compression,block):
    return (np.round(f(block,u,v)/Q(u,v,compression)))


"""
@param
    block : a block represented by an array 
    u,v : position of the point
@return
    value : return the value of block[u][v]
"""
def f(Block,u,v):
    return Block[u][v]

"""
@param
    dctArray : all the dctblock of blocks
    Blocks : base image cut in blocks
    N*M : size of a block
    img_size : size of the base img
    compression : final compression rate 
@return
    table : quantified blocks
"""
def Quantification_table(dctArray,N,M,img_size,compression):
    range_x= int(img_size[0]/N)
    range_y= int(img_size[1]/M)
    blockSize = (range_x,range_y,N,M)
    table = np.zeros(blockSize)
    for i in range(range_x):
        for j in range(range_y):
            for k in range (N):
                for l in range(M):
                    table[i][j][k][l] = Fq(f,k,l,compression,dctArray[i][j])
            #print(table[i][j])
    
    return table


def flq(u,v,compression,block):
    return ((block[u][v])*Q(u,v,compression))
            
def Unquantification(table,N,M,img_size,compression):
    range_x= int(img_size[0]/N)
    range_y= int(img_size[1]/M)
    blockSize = (range_x,range_y,N,M)
    unquantified = np.zeros(blockSize)
    for i in range(range_x):
        for j in range(range_y):
            for k in range (N):
                for l in range(M):
                    unquantified[i][j][k][l] = flq(k,l,compression,table[i][j])
                    
    return unquantified
"""
@param 
    N*M : size of the blocks
    imgPath : local path to the img
@return
    dctArray : a N*M*(img_size[0]/N)*(img_size[1]/M) array corresponding to the dct of each N*M blocks of the picture.
    img_size : size of the base image
    dctblocks : base image cut in N*M blocks
"""
def dctBlocks(N,M,imgPath):
    img=io.imread(imgPath)
    img_gray = rgb2gray(img)
    img_gray=sk.img_as_float(img_gray)
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
            
    return (dctArray,img_size,img_gray)
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
    finalPsnr = np.zeros((range_x,range_y))
    #finalSsim = np.zeros((range_x,range_y))
    for i in range(range_x):
        #   temping = np.zeros((N,M))
        for j in range(range_y):
            tempimg = idct2(dctArray[i][j])
            psnr=measure.compare_psnr(dctArray[i][j],tempimg,1.0)
            #ssim=measure.compare_ssim(dctArray[i][j],tempimg)
            finalPsnr[i][j]=psnr;
            #finalSsim[i][j]=ssim
            #tempimg=np.ubyte(np.round(255.0*tempimg,0))
            for k in range(N):
                for l in range(M):
                    finalImg[i*N + k][j*M +l]=(tempimg[k][l]) 
    return (finalImg,finalPsnr)

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
    fich=open(file_name,'w')
    fich.write(file) 
    fich.close()
    
"""
create a new file called file_name and write inside the content of file in binary mode
@param
    file : the content to write
    file_name : the name of the file that will be created / overwritted
"""
def exportInBinary(file,file_name):
    fich=open(file_name,'wb')
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
    quality : the desired quality for the quantification table
"""

def compression(N,M,imgPath,quality,with_print):
    if(with_print):
        print("\n\n\nfile : " + imgPath + " taille : " +str(N) +" "+str(M))
    (dctArray,img_size,imgGray)=dctBlocks(N,M,imgPath)
    ImgNew = concatDct(dctArray,N,M,img_size)
    #print(ImgNew)
    exportInBinary(ImgNew,"dct"+str(N)+".dat")
    
    table = Quantification_table(dctArray,N,M,img_size,quality)
    exportInBinary(table,"quantified_blocks"+str(N)+".dat")
    unquantified = Unquantification(table,N,M,img_size,quality)
    
    if(with_print):    
        plt.figure(3)
        plt.imshow(np.log(1.0+ImgNew),cmap='gray')
        plt.show()
    
    (finalImg,finalPsnr) =recompression(unquantified,N,M,img_size)
    plt.figure(4)
    
 
    #remettre le type des données ici entre 0 et 255 donc uint8
    if(with_print):
        plt.imshow(finalImg,cmap='gray')
        plt.show()
        
    fich=open('madct'+str(N)+'.dat','wb')
    fich.write(np.reshape(finalImg,-1)) 
    #on étend le tableau en 1D pour pouvoir enregistrer chaque octet
    fich.close()
    
    #pour sauver l'image en format jpeg pour une qualité voulue
    finalImg=pil.Image.fromarray(np.ubyte(np.round(255.0*finalImg,0)))
    finalImg.save('essai'+str(N)+'.jpeg')
    monImlu=io.imread("essai"+str(N)+'.jpeg')
    
    img_gray2 = rgb2gray(monImlu)
    img_gray2=sk.img_as_float(monImlu)
    finalPsnr=measure.compare_psnr(imgGray,img_gray2,quality)
    
    compression = (1.0*img_size[0]*img_size[1]/os.path.getsize("essai"+str(N)+'.jpeg'))
    if(with_print):
        print( "taille= ",os.path.getsize("essai"+str(N)+'.jpeg'), "en octet")
        print("compression =", 1.0*img_size[0]*img_size[1]/os.path.getsize("essai"+str(N)+'.jpeg'))
        print("psnr : "+str(finalPsnr))
        
    return (compression,(finalPsnr))

def ourQualityInfluence(startingValue,maxValue,step,N,M,img):
    compressionArray = np.zeros(int((maxValue-startingValue)/step))
    psnrArray = np.zeros(int((maxValue-startingValue)/step))
    qualityArray = np.zeros(int((maxValue-startingValue)/step))
    count = 0
    for i in range(startingValue,maxValue,step):
        (compression_rate,psnr) = compression(N,M,img,i,0)
        compressionArray[count]=compression_rate
        psnrArray[count]=psnr
        qualityArray[count]=i        
        count=count+1
        
    plt.plot(qualityArray, psnrArray)
    plt.ylabel('quality / psnr')
    plt.show()
    
    plt.plot(compressionArray,psnrArray)
    plt.ylabel('compression_rate / psnr')
    plt.show()
    
def compresJpeg(img,quality):
    monImlu=io.imread(img)
    img_gray = rgb2gray(monImlu)
    img_gray =sk.img_as_float(monImlu)
    
    finalImg=pil.Image.fromarray(np.ubyte(np.round(255.0*img_gray,0)))
    finalImg.save(img+str(2)+'.jpeg',quality=quality)
    
    monImlu2=io.imread(img+str(2)+'.jpeg')
    
    img_gray2 = rgb2gray(monImlu2)
    img_gray2 =sk.img_as_float(monImlu2)
    psnr = measure.compare_psnr(img_gray,img_gray2)
    
    compressionRate = (os.path.getsize(img)/os.path.getsize(img+str(2)+'.jpeg'))
    return (compressionRate,psnr)

    

def jpegQualityInfluence(startingValue,maxValue,step,img):
    compressionArray = np.zeros(int((maxValue-startingValue)/step))
    psnrArray = np.zeros(int((maxValue-startingValue)/step))
    qualityArray = np.zeros(int((maxValue-startingValue)/step))
    compressionArray = np.zeros(int((maxValue-startingValue)/step))
    psnrArray = np.zeros(int((maxValue-startingValue)/step))
    qualityArray = np.zeros(int((maxValue-startingValue)/step))
    count = 0
    for i in range(startingValue,maxValue,step):
        (compression_rate,psnr) = compresJpeg(img,i)
        compressionArray[count]=compression_rate
        psnrArray[count]=psnr
        qualityArray[count]=i        
        count=count+1
        
    plt.plot(qualityArray, psnrArray)
    plt.ylabel('quality / psnr')
    plt.show()
    
    plt.plot(compressionArray,psnrArray)
    plt.ylabel('compression_rate / psnr')
    plt.show()

def compareQualityInfluence():
    print("Influence de la qualite sur le psnr pour NOTRE methode")
    ourQualityInfluence(1,100,1,64,64,"horse.bmp")
    print("Influence de la qualite sur le psnr pour la compression JPEG")
    jpegQualityInfluence(1,100,1,"horse.bmp")

compression(4,4,"horse.bmp",2,1)    
compression(8,8,"horse.bmp",2,1) 
compression(16,16,"horse.bmp",2,1) 
compression(64,64,"horse.bmp",2,1) 

compareQualityInfluence()

"""
Q4 (compression):
    pour n=4:    
        taille de la dct quantifiee compressee : 83 000
        taille de la dct compressee : 1 000 000
        taille de l'image bmp compressee : 199 824
    199824/83000 = 2.4
"""

print("execution ended")
