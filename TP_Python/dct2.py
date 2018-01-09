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