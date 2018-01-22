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
import re
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

"""
@author Charles Marchand - Aymeric Vial-Grelier
"""
   
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


"""
print a plot of quality / PSNR and compression_rate/PSNR using our compression system
@param
    startingValue: The first quality value tested
    maxValue     : The last quality value tested
    step         : The step of quality
    N*M          : Size of blocks
    img          : The img to test with
"""
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
    
"""
compress a specified img with a speficied quality using jpeg librairy and return the compression rate and the psnr
@param
    img     : the img to compress
    quality : the quality applyied to the img
@return
    compression_ rate : the value of the compression rate
    psnr              : the value of the psnr
"""    
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

    
"""
print a plot of quality / PSNR and compression_rate/PSNR using jpeg compression system
@param
    startingValue: The first quality value tested
    maxValue     : The last quality value tested
    step         : The step of quality
    img          : The img to test with
"""
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

"""
compare Quality influence on PSNR and compression rate
@param
    our  : true if you want to plot our method plots
    jpeg : true if you want to plot jpeg method plots
"""
def compareQualityInfluence(our,jpeg):
    if(our):
        print("\n\n\nInfluence de la qualite sur le psnr pour NOTRE methode")
        ourQualityInfluence(1,100,1,64,64,"horse.bmp")
    if(jpeg):
        print("\n\n\nInfluence de la qualite sur le psnr pour la compression JPEG")
        jpegQualityInfluence(1,100,1,"horse.bmp")

"""
return the name of the desired picture
@param
    name : the name of the picture series
    number : the number of the desired name
    ext : the extension of the picture series
@return
    n : name+number+'.'+ext
"""
def getName(name,number,ext):
    n=name
    if(number<10):
        n=n+str(0)+str(number)
    else:
        n=n+str(number)
    return n+"."+ext

"""
return the float rgb2gray of the img
@param
    name : the name of the img to deal with
@return
    gray : the float rgb2gray of name img
"""
def getGray(name):
    img= io.imread(name)
    gray= rgb2gray(img)
    gray = sk.img_as_float(gray)
    return gray

"""
return the float rgb2gray of an img designed by a serie name, a number and it extension
@param 
    name : name of the picture
    number : the number of the picture in it serie
    ext : the ext of the series
@return
    gray : the float rgb2gray of the picture
"""
def getGrayfromNumber(name,number,ext):
    return getGray(getName(name,number,ext))

"""
return true if value1=value2
@param
    value1 : the first value
    value2 : the second value
@return
    bool : true if value1==value2 else false
"""
def eq(value1,value2):
    return (value1==value2)


"""
return a movement vector between img0 and img1 using the fun comparaison method
@param
    img0 : the first img
    img1 : the second img
    fun : the comparaison method
@return
    potentialVector : the deplacement vector between img0 and img1
"""
def calculVecteurDeplacement(img0,img1,fun):
    img_size = img0.shape
    potentialVector = []
    count=0
    for i in range(0,img_size[0]):
        for j in range(0,img_size[1]):
            if(not(img1[i][j]==img0[i][j])):
                b = 1
                for k in range(1,16):
                    for l in range (1,16):
                        if((i-k)>0 and b):
                            if((j-l)>0):
                                if(fun(img1[i-k][j-l],img0[i][j])):
                                    potentialVector.append([])
                                    potentialVector[count].append(i)
                                    potentialVector[count].append(j)
                                    potentialVector[count].append(i-k)
                                    potentialVector[count].append(j-l)
                                    potentialVector[count].append(k+l)
                                    count=count+1
                                    b = 0
                            if((j+l)<img_size[1]):
                                if(fun(img1[i-k][j+l],img0[i][j])):
                                    potentialVector.append([])
                                    potentialVector[count].append(i)
                                    potentialVector[count].append(j)
                                    potentialVector[count].append(i-k)
                                    potentialVector[count].append(j+l)
                                    potentialVector[count].append(k+l)
                                    count=count+1
                                    b = 0
                        if((i+k)<img_size[0] and b):
                            if((j-l)>0):
                                if(fun(img1[i+k][j-l],img0[i][j])):
                                    potentialVector.append([])
                                    potentialVector[count].append(i)
                                    potentialVector[count].append(j)
                                    potentialVector[count].append(i+k)
                                    potentialVector[count].append(j-l)
                                    potentialVector[count].append(k+l)
                                    count=count+1
                                    b = 0 
                            if((j+l)<img_size[1]):
                                if(fun(img1[i+k][j+l],img0[i][j])):
                                    potentialVector.append([])
                                    potentialVector[count].append(i)
                                    potentialVector[count].append(j)
                                    potentialVector[count].append(i+k)
                                    potentialVector[count].append(j+l)
                                    potentialVector[count].append(k+l)
                                    count=count+1
                                    b = 0 
    return potentialVector

"""
find in tab the value and return it index
@param
    value : the value to look for
    tab : the tab to look in
@return
    i : the index of the value
returns -1 if nothing found
"""
def find(value,tab):
    for i in range(0,len(tab)):
        if(tab[i][2]==value[0] and tab[i][3]==value[1]):
            return i
    return -1
"""
combine and img lastImg and apply the vector movement on it to predict the next img
@param
    lastImg : the img to deal with
    movement : the vector to apply on img
@return
    return the potential next img
"""
def CombineNext(lastImg,movement):
    newImg = np.copy(lastImg)
    img_size = lastImg.shape
    for i in range(0,len(movement)):
        deltaX = np.abs(movement[i][0]-movement[i][2])
        deltaY = np.abs(movement[i][1] - movement[i][3])
        coef = deltaX / deltaY
        lastImg[movement[i][2]][movement[i][3]]
        newValX = int( (movement[i][4]*(coef))+movement[i][4] )
        newValY = int( (movement[i][4]*(1/coef))+movement[i][3] )
        if(newValX>=0 and newValX<img_size[0] and newValY>=0 and newValY<img_size[1]):
            newImg[newValX][newValY] = lastImg[movement[i][2]][movement[i][3]]
        
    return newImg
"""
define the next vector based on multiples vector
@param
    MovementList : a list of vector
@return
    validList : A predicted movement vector based on past vectors 
"""
def defineNext(MovementList):
    size = len(MovementList)
    validList=[]
    print("tentative de prédiction avec "+str(size)+" vecteurs")
    if(size>0):
        valid = []
        for i in range(1,size):
            #each VectorList
            if(valid==[]):
                print("calcul des déplacements continus entre 0 et " + str(i+1))
                valid = ValidateMovement(MovementList[i-1],MovementList[i])
            else:
                print("calcul des déplacements continus entre 0 et " + str(i+1))
                valid = ValidateMovement(valid,MovementList[i])
            validList.append(valid)
    return validList

"""
Look for continious mouvement between vector1 and vector2
@param
    vector1 : the first vector
    vector2 : the second vector
@return
    valid : a list of vector which are continious between vector1 and vector2
"""        
def ValidateMovement(vector1,vector2):
    valid = []
    count = 0
    for i in range(0,len(vector2)):
        value = vector2[i]
        index =find(value,vector1)
        if(index != -1):
            valid.append([])
            valid[count].append(vector2[i][0])
            valid[count].append(vector2[i][1])
            valid[count].append(vector2[i][2])
            valid[count].append(vector2[i][3])
            valid[count].append((vector2[i][4]+vector1[index][4])/2)    
            count=count+1
        
    print("Mouvements continus : "+str(len(valid)))
    return valid

"""
predict the 2-imgCount last img of a series defined by a name and an extension
@param
    name : the name of the img serie
    ext : the extension of the img serie
    imgCount : 2-imgCount last img will be predicted
    
imgCount should be >2
"""
def prediction(name,ext,imgCount):
    if(imgCount<2):
        print("Nombre d'images insuffisant pour réaliser une prédiction")
        return ;
    imgList = []
    imgList.append(getGrayfromNumber(name,0,ext))
    img1 = getGrayfromNumber(name,1,ext)
    imgList.append(img1)
    Vector = calculVecteurDeplacement(imgList[0],img1,eq)
    VectorList = []
    VectorList.append(Vector)
    print("Mouvements détectés entre 0 et 1 : " + str(len(Vector)))
    i=2
    for i in range(2,imgCount):
        nextImg = getGrayfromNumber(name,i,ext)
        imgList.append(nextImg)
        Vector = calculVecteurDeplacement(imgList[i-1],nextImg,eq)
        VectorList.append(Vector)
        print("Mouvements détectés entre "+str(i-1)+" et " +str(i)+" : " + str(len(Vector)))
    
    nexts=defineNext(VectorList) 
    for i in range(0,len(nexts)):
        print("prediction de " + getName(name,i+3,ext) + " : ")
        imgNPred = CombineNext(imgList[i+2],nexts[i])
        plt.imshow(imgNPred,cmap='gray')
        plt.show()
        
"""
Our BEAUTIFUL and PRETTY presentation function
"""        
def presentation():
    print("\n\nBonjour, voici le TP réalisé par Charles Marchand et Aymeric Vial-Grelier")
    print("Nous allons vous montrer notre calcul de DCT ainsi que la compression d'une image")
    b=1
    while(b):        
        print("\n\nMerci d'indiquer ce que vous voulez sous le format N-M-fichier-compression ")
        print("Valeurs de base : 8-8-horse.bmp-2, \nEntrez 0 pour utiliser cette configuration")
        print("Entrez 1 pour passer à l'étape suivante")
        reponse = input()
        if(len(reponse)>2):
            tab=reponse.split("-")
            if(len(tab)==4):
                compression(int(tab[0]),int(tab[1]),tab[2],int(tab[3]),1)
            else:
                print("mauvais format , exemple : 8-8-horse.bmp-2 ")
        else:
            value = int(re.search(r'\d+', reponse).group())
            if(value==0):
                compression(8,8,"horse.bmp",2,1) 
            elif(value==1):
                b=0
    
    print("Réponse a quelque questions :")
    print("1.Visualisez l'image DCT, quelle conclusion pouvez-vous faire sur la localisation et la répartition des données?")
    print("On peut voir dans la DCT que l'information est répartie majoritairement en haut à gauche de la DCT")
    print("C'est ici qu'on peut voir les fréquences faible.")
    
    
    print("\n\n2.D’après vous pourquoi le groupe jpeg a choisi la taille 8 ?")
    print("Comme l'on peut voir en faisant plusieurs test avec les fonctions de compression, les blocs de tailles 8*8,")
    print("Donnes un rendu de l'images beaucoup plus satisfant que les autres :")
    print(" - Les zones dégradés semblent plus jolie que lors des 4*4")
    print(" - On évite une grande partie de l'effet mousitiques des plus grosses séparation")
    
    print("\n\n4.Calculez le taux de compression")
    print("Pour vérifier les informations suivante regarder le .rar TP_python.rar")
    print(" pour n=8, qualité=2")
    print("   - taille de la dct quantifiee compressee : 80 000o")
    print("   - taille de la dct compressee : 1 884 000o")     
    print("   - taille de l'image bmp compressee : 200 000o")
    print(" taux de compression = 2.5 ")
        
    print("\n\n Appuyer sur entrée pour continuer")
    input()
    compareQualityInfluence(0,1)
    print("5.Conclusion sur ces deux indicateurs ?")
    print("On peut voir que plus l'on compression l'image plus les indicateurs sont faible")
    print("aussi plus on augmente la qualité plus le PSNR augmente.")
    print("Ces indicateurs on donc l'air fiable et représente bien la qualitée réelle d'une image")
    print("\n\n Appuyer sur entrée pour continuer")
    input()
    b=1
    print("Nous allons maintenant vous présenter notre algorithme de prédiction d'image")
    while(b):
        print("\n\nMerci d'indiquer le numéro de l'image à prédire")
        print("compris entre 3 et 20. Afin de quitter le programme merci d'entrer 0")
        print("Attention, cela risque d'être long si le nombre est grand, cependant la qualitée de la prédiction augmentera.")
        reponse = input()
        value = int(re.search(r'\d+', reponse).group())
        if(value == 0):
            b=0
        elif(value >= 2 and value<=20):
            prediction("taxi_","bmp",value)
    print("\n\nMerci, d'avoir suivis notre compte rendu.")
            

presentation()
print("\nExecution ended, press enter to exit")
input()
