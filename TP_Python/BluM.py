# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:56:06 2016

@author: ladretp
"""
import numpy as np
import skimage as sk
from skimage import color, data,feature,filters
from scipy.ndimage import convolve 
       
""" Mesure de Flou
     Méthode Fredérique Crêté
[1]: The blur effect: perception and estimation with a new no-reference
 perceptual blur metric . Frederique Crete, Thierry Dolmiere, Patricia Ladret,
 Marina Nicolas.Electronic Imaging 2007. 64920I-64920I-11. 
 International Society for Optics and Photonics """
 
def DiffCol(I):
    taille=np.shape(I)
    S=np.zeros(taille,dtype='float')
    S=np.abs(I[:,2:taille[1]]-I[:,1:taille[1]-1])
    S[:,0]=S[:,1]
    return S
def Difflig(I):
    taille=np.shape(I)
    S=np.zeros(taille,dtype='float')
    S=np.abs(I[2:taille[0],:]-I[1:taille[0]-1,:])
    S[0,:]=S[1,:]
    return S
    
def BluM(I):
   
    h=np.ones((11,1),dtype=np.float)/11
   #print(h)
    #h[2,:]=np.array([1.0/5,1.0/5,1.0/5,1.0/5,1.0/5])
    taille=np.shape(I)
    
    if np.size(taille)==3:
        I=color.rgb2gray(I)
        
    y=np.copy(I)
    y=sk.img_as_float(I)
  
    #Flou Vertical
    Iver=convolve(y,h)

    ImNette=DiffCol(y)
    
    
    ImFlou=DiffCol(Iver)
    T=np.abs(ImNette-ImFlou)
    M1=np.sum(ImNette[2:taille[0]-1,2:taille[1]-1])
    M2=np.sum(T[2:taille[0]-1,2:taille[1]-1])
    F1=np.abs((M1-M2))/M1
    
    #Flou Horizontal
    Ihor=convolve(y,np.transpose(h))
    
    ImNette=np.copy(I)
    ImNette=sk.img_as_float(ImNette)
    ImNette=DiffCol(y)
    

    ImFlou=DiffCol(Ihor)
    T=np.abs(ImNette-ImFlou)
    M1=np.sum(ImNette[2:taille[0]-1,2:taille[1]-1])
    M2=np.sum(T[2:taille[0]-1,2:taille[1]-1])
    F2=np.abs((M1-M2))/M1
    F= np.array([F1,F2])
    return np.amax(F)
    
#________________________________________________________________________
# Mesure de bloc méthode de Perra
"""  [2]Estimating blockness distortion for performance evaluation 
of picture coding algorithms. D.D. Giusto ; M. Perra. 
1997 IEEE Pacific Rim Conference on Communications, Computers and Signal Processing, PACRIM.
 """ 
def BlocM(I):
    taille=np.shape(I)
    taillex=taille[0]
    tailley=taille[1]
    
    if np.size(taille)==3:
        I=color.rgb2gray(I)
    y=np.copy(I)
    y=sk.img_as_float(I)
    h=np.zeros((3,3),dtype=np.float)
    h[1,:]=np.array([-1.0,0.0,1.0,])
    Dx=convolve(y,h)
    Dy=convolve(y,np.transpose(h))
    D=np.sqrt(Dx*Dx+Dy*Dy)
    s1=0.0;
    s2=0.0;nbbloc=0.0;MES_Bloc=0.0;
    for k in np.arange(8,taillex-24,8):
        for l in np.arange(8,tailley-24,8):
            BH1=Dy[k,l:l+8]
            BH2=Dy[k+8,l:l+8]
            BV1=Dx[k:k+8,l]
            BV2=Dx[k:k+8,l+8]
            BC=D[k+1:k+7,l+1:l+7]
            omega1H=np.array([BV1, BV2]);
            omega1V=np.array([BH1, BH2]);
            valabs1=np.amax(np.abs(omega1H));
            valabs2=np.amax(np.abs(omega1V));
            s11=0.0;s12=0.0;
            if (valabs1 > 0.0):
                s11=np.sum((np.abs(omega1H)/np.amax((np.abs(omega1H)))));
                
        
            if (valabs2 > 0.0):
                s12=np.sum((np.abs(omega1V)/np.amax((np.abs(omega1V)))));
                
            s1=(s11+s12)/32.0;
            
            if np.amax(np.abs(BC))>0.0:
                s2=np.sum(BC)/np.amax(BC)/36.0;
                if (s1*s1+s2*s2)>0.0:
                    MES_Bloc= MES_Bloc+np.abs((s1*s1-s2*s2)/(s1*s1+s2*s2));
                    nbbloc=nbbloc+1.0;
            
            else:
                if ((valabs1>0.0) | (valabs2>0.0)):
                    MES_Bloc=MES_Bloc+1.0;
                
                nbbloc=nbbloc+1;

    MES_Bloc=MES_Bloc/nbbloc;

    
    
    return MES_Bloc;
        
     