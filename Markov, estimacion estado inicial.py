# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:40:51 2020

@author: Media Service
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as S
import scipy.optimize as opt
import scipy.sparse as sp

M0 = S.Matrix([[0.99,0.07,0,0,0,0,0],
               [0.004,0.90,0.049,0,0,0,0],
               [0.001,0.01,0.95,0,0,0,0],
               [0.005,0,0,0.96,0.02,0,0],
               [0,0.02,0,0.034,0.961,0.061,0],
               [0,0,0.001,0.003,0.015,0.939,0],
               [0,0,0,0.003,0.004,0,1]])

sl = S.symbols('sl')
sc = S.symbols('sc')
sa = S.symbols('sa')
il = S.symbols('il')
ic = S.symbols('ic')
ia = S.symbols('ia')
c = S.symbols('c')

estado = S.Matrix([sl,sc,sa,il,ic,ia,c])

#en este caso dejamos como incógnitas sl,il y c
estado = estado.subs([(sa,0),(sc,0),(ia,0),(ic,0)])

for i in range(400):
    estado = M0*estado

print(estado) #devuelve un vector de 7 polinomios

coeficientes = np.zeros((7,3),dtype=np.float64)
for i in range(7): #matriz 7 x 3 donde almacenamos los coeficientes de los polinomios
    coef = S.poly(estado[i])
    print(coef)
    coeficientes[i,3-np.size(coef.coeffs()):] = coef.coeffs()

#print(coeficientes)

coefsanos = np.sum(coeficientes[:3,:],axis=0)
coefinfect = np.sum(coeficientes[3:6,:],axis=0)
coefcurados = coeficientes[-1,:]

matriz = np.zeros((3,3)) #matriz 3 x 3 con los coefs del sistema de ecuaciones
matriz[0,:] = coefsanos #la rellenamos
matriz[1,:] = coefinfect
matriz[2,:] = coefcurados

#print(matriz)

b = np.array([0.02,0.38,0.6]) #término independiente del sistema
solucion1 = opt.nnls(matriz,b) #solucion por nnls
solucion3 = np.linalg.solve(matriz,b) #solucion por solve
#print(solucion1[0])
#print(solucion3)
#print(np.abs(np.dot(matriz,solucion1[0])-b)) #error por nnls
#print(np.abs(np.dot(matriz,solucion3)-b)) #error por solve






    




            


    