# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:02:58 2020

@author: Media Service
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as S

M0 = S.Matrix([[0.99,0.07,0,0,0,0,0,0],             #matriz para la estimación
               [0.004,0.90,0.049,0,0,0,0,0],        #del estado inicial
               [0.001,0.01,0.95,0,0,0,0,0],
               [0.005,0,0,0.96,0.02,0,0,0],
               [0,0.02,0,0.034,0.961,0.061,0,0],
               [0,0,0.001,0.003,0.015,0.939,0,0],
               [0,0,0,0.003,0.004,0,1,0],
               [0,0,0,0,0,0,0,1]])

M1 = S.Matrix([[0.99,0.07,0,0,0,0,0,0],        #matriz CASO 2          
               [0.004,0.90,0.049,0,0,0,0,0],        
               [0.001,0.01,0.959,0,0,0,0,0],
               [0.005,0,0,0.959,0.02,0,0,0],
               [0,0.02,0,0.034,0.9609,0.06,0,0],
               [0,0,0.001,0.003,0.015,0.938,0,0],
               [0,0,0,0.003,0.004,0,1,0],
               [0,0,0,0.001,0.0001,0.002,0,1]])

M2 = S.Matrix([[0.99,0.07,0,0,0,0,0.001,0],         #matriz CASO 3
               [0.004,0.90,0.049,0,0,0,0,0],
               [0.001,0.01,0.95,0,0,0,0,0],
               [0.005,0,0,0.959,0.02,0,0,0],
               [0,0.02,0,0.034,0.9609,0.06,0,0],
               [0,0,0.001,0.003,0.015,0.938,0,0],
               [0,0,0,0.003,0.004,0,0.999,0],
               [0,0,0,0.001,0.0001,0.002,0,1]])

M3 = S.Matrix([[0.9944,0.1,0,0,0,0,0.001,0],        #matriz CASO 4
               [0.0005,0.5795,0.05,0,0,0,0,0],
               [0.0001,0.0005,0.90,0,0,0,0,0],
               [0.005,0,0,0.9615,0.04,0,0,0],
               [0,0.32,0,0.032,0.944,0.06,0,0],
               [0,0,0.05,0.0025,0.01,0.935,0,0],
               [0,0,0,0.003,0.001,0,0.999,0],
               [0,0,0,0.001,0.005,0.005,0,1]])

M4 = S.Matrix([[0.991,0.07,0,0,0,0,0.001,0],         #matrices CASO 5
               [0.004,0.902,0.049,0,0,0,0,0],
               [0.001,0.01,0.9502,0,0,0,0,0],
               [0.004,0,0,0.9573,0.02,0,0,0],
               [0,0.018,0,0.034,0.95494,0.06,0,0],
               [0,0,0.0008,0.003,0.015,0.93985,0,0],
               [0,0,0,0.005,0.01,0,0.999,0],
               [0,0,0,0.0007,0.00006,0.00015,0,1]])

M5 = S.Matrix([[0.992,0.07,0,0,0,0,0.001,0],
               [0.004,0.905,0.049,0,0,0,0,0],
               [0.001,0.01,0.9503,0,0,0,0,0],
               [0.003,0,0,0.9546,0.02,0,0,0],
               [0,0.015,0,0.034,0.94497,0.06,0,0],
               [0,0,0.0007,0.003,0.015,0.9399,0,0],
               [0,0,0,0.008,0.02,0,0.999,0],
               [0,0,0,0.0004,0.00003,0.0001,0,1]])

#lo hacemos en simbólico para que trabaje con las incógnitas como si fueran números
#y para poder sustituir los valores cuando queramos
sl = S.symbols('sl')
sc = S.symbols('sc')
sa = S.symbols('sa')
il = S.symbols('il')
ic = S.symbols('ic')
ia = S.symbols('ia')
c = S.symbols('c')
m = S.symbols('m')

estado = S.Matrix([sl,sc,sa,il,ic,ia,c,m]) #estado inicial

estado = estado.subs([(sl,0.995),(sc,0),(sa,0),(il,0.005),(ic,0),(ia,0),(c,0),(m,0)])
#en el apartado de estimación del estado inicial hay que quitar la última componente de muertos
 
N = 1500 #tiempo que dura la simulación

rep = np.zeros((N,4)) #para representar
X = np.arange(N)

for i in range(N): #aquí iteramos. Vamos cambiando las condiciones en función de en qué caso estemos
    #if estado[4] + estado[5] > 0.2:
    #    print(i)
    #estado = M0*estado
    #else:
    #if i > 100:
    estado = M1*estado
            
                #estado = M2*estado
        #    elif i>600 and i<1000:
        #        estado = M4*estado
    #else: 
     #   estado = M2*estado
            
    rep[i,0] = estado[0] + estado[1] + estado[2] #SL + SC + SA
    rep[i,1] = estado[3] + estado[4] + estado[5] #IL + IC + IA
    rep[i,2] = estado[6] #Cu
    rep[i,3] = estado[7] #M

#representamos
plt.plot(X,rep[:,0],label='sanos',color='g')
plt.plot(X,rep[:,1],label='infect',color='r')
plt.plot(X,rep[:,2],label='curados',color='b')
plt.plot(X,rep[:,3],label='muertos',color='k')
plt.title('Caso 3')
#plt.xticks(np.array([0,400])) 
#plt.yticks(np.array([0,0.02,0.38,0.5,0.6]))
plt.xlabel('Nº pasos')
plt.ylabel('Porcentaje')
plt.legend()
plt.grid()
plt.show()

#representamos las derivadas
derivada1 = np.diff(rep[:,1])/np.diff(X)
plt.plot(X[1::5],derivada1[::5],'r')
plt.title('Derivada infectados, caso 3')
plt.xlabel('Nº pasos')
plt.ylabel('Porcentaje/Nº pasos')
plt.grid()
plt.show()

derivada2 = np.diff(rep[:,3])/np.diff(X)
plt.plot(X[1::5],derivada2[::5],'k')
plt.title('Derivada muertos, caso 2')
plt.xlabel('Nº pasos')
plt.ylabel('Porcentaje/Nº pasos')
plt.grid()
plt.show()

      